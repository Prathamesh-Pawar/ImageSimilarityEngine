import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tensorflow.python.keras.testing_utils import numeric_test
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from collections import defaultdict
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import os

from ImageSimilarityEngine.SimpleClassifier.cnn_network import ConvCNN
from ImageSimilarityEngine.Triplet.networks import TripletNet, EmbeddingNet

if __name__ == "__main__":
    # Create test image data directory
    output_dir = "test_data"
    os.makedirs(output_dir, exist_ok=True)

    # Define function to denormalize image.
    def denormalize(tensor, mean, std):
        mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
        return tensor * std + mean  # Reverse normalization

    # Define transform to convert tensor to image
    to_pil = transforms.ToPILImage()


    # Saving function for test and similar images
    def save_images(test_idx, test_image, similar_images, test_dir, category):
        # Subdirectory for the test image
        print("Saving...")
        test_image_dir = os.path.join(test_dir, f"test_image_{test_idx}")
        os.makedirs(test_image_dir, exist_ok=True)

        # Save the test image
        test_image = test_image.squeeze(0)
        test_image = denormalize(test_image.cpu(), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        test_image_path = os.path.join(test_image_dir, "test_image.png")
        to_pil(test_image.clamp(0, 1)).save(test_image_path)

        # Subdirectory for the category (feature_map/class_prediction)
        category_dir = os.path.join(test_image_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        # Save the similar images
        for i, (_, img_tensor) in enumerate(similar_images):
            img_tensor = img_tensor.squeeze(0)
            img_tensor = denormalize(img_tensor.cpu(), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            similar_image_path = os.path.join(category_dir, f"similar_image_{i + 1}.png")
            to_pil(img_tensor.clamp(0, 1)).save(similar_image_path)


    # Download the NLTK stopwords if not already
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')

    # Prepare captioning model and device to process
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Recommended online for torch
    kw_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    kw_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


    # Load models for custom image searching and simple CNN classification
    model_custom = TripletNet(EmbeddingNet())
    model_custom.load_state_dict(torch.load('best_model_triplet.pth'))
    model_simple = ConvCNN()
    model_simple.load_state_dict(torch.load('best_model_classifier.pth'))
    model_custom.eval()
    model_simple.eval()

    # Define CIFAR-10 dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cifar10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(cifar10, batch_size=64, shuffle=True)

    # Store images for each class in dictionary
    class_dict = defaultdict(list)

    # Iterate through dataset and collect 50 images per class
    for inputs, labels in test_loader:
        for i, label in enumerate(labels):
            if len(class_dict[label.item()]) < 50:
                class_dict[label.item()].append((inputs[i], label.item()))
        if all(len(class_dict[label.item()]) >= 50 for label in labels):
            break


    # Define function for getting feature map from custom model
    def get_feature_map(model, input_image):
        return model.get_embedding(input_image)

    # Gets class prediction from SimpleCNN
    def get_class_prediction(model, input_image):
        output = model(input_image)
        _, pred_class = torch.max(output, 1)
        return pred_class.item()

    # Generate keywords for image
    def generate_keywords(gen_img):
        denorm_img = denormalize(gen_img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        denorm_img = denorm_img.clamp(0, 1)

        # Convert to NumPy array (H, W, C) and rescale to 0–255
        denorm_img = denorm_img.permute(1, 2, 0).numpy()  # Shape: (H, W, C)
        denorm_img = (denorm_img * 255).clip(0, 255).astype("uint8")

        # Generate interpretable image
        kw_img = Image.fromarray(denorm_img)

        # Prepare the processed input for the image
        kw_inputs = kw_processor(images=kw_img, return_tensors="pt")

        # Generate the caption for the image
        out = kw_model.generate(**kw_inputs)

        # Decode the generated caption
        caption = kw_processor.decode(out[0], skip_special_tokens=True)

        # Tokenize caption
        words = word_tokenize(caption)

        # Get stopwords
        stop_words = set(stopwords.words('english'))

        # Perform part of speech tagging
        tagged_words = pos_tag(words)

        # Filter out stopwords and non-nouns/adjectives
        kw = [
            word for word, tag in tagged_words
            if word.lower() not in stop_words and (tag.startswith('NN') or tag.startswith('JJ'))
        ]

        # Limit the result to a maximum 5 keywords
        return kw[:5]


    # Store feature maps and predictions
    feature_maps = {}
    class_predictions = {}
    img_keywords = {}

    # Generate dictionary for all images.
    for class_label, images in class_dict.items():
        for idx, (image, _) in enumerate(images):
            image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
            feature_map = get_feature_map(model_custom, image)
            class_prediction = get_class_prediction(model_simple, image)
            keywords = generate_keywords(image.squeeze(0))

            feature_maps[(class_label, image)] = feature_map
            class_predictions[(class_label, image)] = class_prediction
            img_keywords[(class_label, image)] = keywords


    # Generate test images
    cifar10_list = list(cifar10)
    random.seed(100)
    num_test = 10
    # Randomly sample images
    test_images = random.sample(cifar10_list, num_test)

    # Extract features and predictions for random images
    test_feature_maps = {}
    test_class_predictions = {}
    test_keywords = {}

    # Generate feature maps and predicted classes for test images.
    for idx, (image, _) in enumerate(test_images):
        image = image.unsqueeze(0).to(device)
        feature_map = get_feature_map(model_custom, image)
        class_prediction = get_class_prediction(model_simple, image)
        keywords = generate_keywords(image.squeeze(0))

        test_feature_maps[idx] = feature_map
        test_class_predictions[idx] = class_prediction
        test_keywords[idx] = keywords

    # Initialize the class_matches dictionary to store predicted classes
    class_matches = defaultdict(list)

    # Match class predictions randomly from dictionary
    for (test_image_idx, predicted_class) in test_class_predictions.items():
        # Iterate over images in the class_predictions dictionary
        for (class_label, img_idx), class_pred in class_predictions.items():
            if predicted_class == class_pred:
                class_matches[predicted_class].append(img_idx)

    # Init dictionary for randomly selected images based on predicted class
    selected_class_images_dict = {}
    for test_image_idx, predicted_class in test_class_predictions.items():
        matched_images = [
            (class_label, img)  # Storing both predicted class and the image tensor
            for (class_label, img), class_pred in class_predictions.items()
            if class_pred == predicted_class
        ]
        # Sample 10 images with matching class predictions
        if len(matched_images) >= 10:
            selected_class_images_dict[test_image_idx] = random.sample(matched_images, 10)
        else:
            selected_class_images_dict[test_image_idx] = matched_images

    # Dictionary for images with the closest feature maps
    closest_images_dict = {}

    # Flatten all feature maps for distance computation
    all_feature_maps = np.array(
        [feature_map.view(-1).detach().cpu().numpy() for (class_label, img_idx), feature_map in
         feature_maps.items()]
    )
    test_feature_map_flat = np.array(
        [feature_map.view(-1).detach().cpu().numpy() for img_idx, feature_map in
         test_feature_maps.items()]
    )

    # Calculate Euclidean distances between test feature maps and all images' feature maps
    distances = euclidean_distances(test_feature_map_flat, all_feature_maps)

    # Map test images to their closest images
    for i, test_image_idx in enumerate(test_keywords.keys()):
        closest_indices = np.argsort(distances[i])[:10]  # Get 10 closest images
        closest_images_dict[test_image_idx] = [list(feature_maps.keys())[index] for index in
                                               closest_indices]

    # Save test images and their similar images for both models
    for test_image_idx, (image, _) in enumerate(test_images):
        image = image.unsqueeze(0).to(device)  # Add batch dimension

        # Save test and feature_map-based similar images
        feature_map_similar_images = closest_images_dict[test_image_idx]
        save_images(
            test_image_idx,
            image,
            feature_map_similar_images,
            output_dir,
            category="feature_map"
        )

        # Save test and class_prediction-based similar images
        class_prediction_similar_images = selected_class_images_dict[test_image_idx]
        save_images(
            test_image_idx,
            image,
            class_prediction_similar_images,
            output_dir,
            category="class_prediction"
        )

    # Calculate overlap in keywords
    def calculate_keyword_overlap(test_keys, t_keywords, similar_image_sets, all_keywords):
        def overlap_for_image(rand_key, similar_keys):
            # Get test image keywords
            rand_keywords = set(t_keywords[rand_key])
            overlap_count = 0
            total_count = 0

            print("New Test Image")
            print(rand_keywords)

            # Check against similar image keywords
            for similar_img in similar_keys[rand_key]:
                similar_keywords = set(all_keywords[similar_img])
                overlap_count += len(rand_keywords.intersection(similar_keywords))
                total_count += len(similar_keywords)
                print(similar_keywords)
                print( len(rand_keywords.intersection(similar_keywords)) / len(similar_keywords) if total_count > 0 else 0)
            return overlap_count / total_count if total_count > 0 else 0

        overlap_scores = {}

        # Iterate over test image keys
        for random_key in test_keys:
            print("Feature Map Eval")
            feature_map_overlap = overlap_for_image(random_key, similar_image_sets['feature_map'])
            print("Simple Classifer Eval")
            class_prediction_overlap = overlap_for_image(random_key, similar_image_sets['class_prediction'])

            # Store overlap scores
            overlap_scores[random_key] = (feature_map_overlap, class_prediction_overlap)

        return overlap_scores

    sim_image_sets = {
        'feature_map': closest_images_dict,  # Images with the closest feature maps
        'class_prediction': selected_class_images_dict  # Images with matching class predictions
    }

    # Calculate keyword overlap metrics for each test image
    eval_overlap = calculate_keyword_overlap(
        test_keys=test_keywords.keys(),
        t_keywords=test_keywords,
        similar_image_sets=sim_image_sets,
        all_keywords=img_keywords
    )

    total_feature_score = 0.0
    total_class_score = 0.0
    feature_better = 0
    # Print the overlap scores for each random image
    for image, (feature_map_score, class_prediction_score) in eval_overlap.items():
        total_feature_score += feature_map_score
        total_class_score += class_prediction_score
        if feature_map_score >= class_prediction_score:
            feature_better += 1
        print(f"Image: {image}")
        print(f"Feature Map-based Keyword Overlap: {feature_map_score: .5f}")
        print(f"Class Prediction-based Keyword Overlap: {class_prediction_score: .5f}")
        print("=" * 50)
    avg_feature_score = total_feature_score / num_test
    avg_class_score = total_class_score / num_test
    print(f"Overall Average Score for Feature Map Based Model: {avg_feature_score:.5f}")
    print(f"Count of Better Performed Trials Feature Map Based Model: {feature_better}")
    print(f"Overall Average Score for Classifier Based Model: {avg_class_score:.5f}")
    print(f"Count of Better Performed Trials Classifier Based Model: {num_test - feature_better}")