import os

# ✅ Change this path to your folder
folder_path = r"unknown\Wolf"

# Replace spaces in folder name to avoid path issues
folder_path = folder_path.replace("\\", "/")

# Get all image files in that folder
image_files = sorted(os.listdir(folder_path))

# Loop through and rename
for idx, filename in enumerate(image_files, start=1):
    ext = os.path.splitext(filename)[1]  # File extension (.jpg, .png, etc.)
    new_name = f"wolf_{idx}{ext}"
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    
    os.rename(old_path, new_path)
    print(f"Renamed: {filename} → {new_name}")

print("✅ All files renamed successfully!")
