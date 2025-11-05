from bing_image_downloader import downloader
import os
import time
import random

categories = [
    "Monkey", "Snake", "Rabbit", "Fox", "Wolf", "Pig", "Goat", "Sheep",
    "Crocodile", "Penguin", "Mouse", "Whale", "Shark", "Camel",
    "Forest", "Mountain", "River", "Sky", "Flower", "Desert",
    "Car", "Bus", "Bicycle", "Train", "Building", "Bridge", "Road",
    "Computer", "Phone", "Furniture", "Clothes", "Bag",
    "Fruit", "Food", "Drink",
    "Person", "Hand", "Eye",
    "Cartoon Animal", "Plush Toy", "Animal Statue", "Animal Painting"
]

output_dir = "unknown"

for idx, category in enumerate(categories, 1):
    print(f"\n📥 ({idx}/{len(categories)}) Downloading: {category}")
    folder_name = category.replace(" ", "_")
    try:
        downloader.download(
            category,
            limit=10,                # number of images per category
            output_dir=output_dir,
            adult_filter_off=True,
            force_replace=False,
            timeout=60,
            verbose=False
        )
        print(f"✅ Downloaded {category}")
    except Exception as e:
        print(f"⚠️ Error downloading {category}: {e}")
    time.sleep(random.randint(2, 6))  # gentle pause
print("\n🎉 All unknown images downloaded successfully!")
