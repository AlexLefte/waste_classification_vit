import os
import shutil
import cv2 

def rename_images_in_folder(folder_path, output_folder):  
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get categories
    categories = os.listdir(input_folder)

    for category in categories: 
        category_path = os.path.join(folder_path, category)
        category_output_path = os.path.join(output_folder, category)
        os.makedirs(category_output_path, exist_ok=True)
        for image_index, file in enumerate(os.listdir(category_path)):
            old_file_path = os.path.join(folder_path, category, file)

            # # Check for transparent pixels
            # # Verifică dacă există un canal alfa
            # img = cv2.imread(old_file_path, cv2.IMREAD_UNCHANGED)
            # if img.shape[-1] == 4:
            #     # Creează o mască pentru pixelii transparenti
            #     alpha_channel = img[:, :, 3]
            #     transparent_mask = alpha_channel == 0

            #     # Înlocuiește pixelii transparenti cu albi
            #     img[transparent_mask, :3] = 255

            img = cv2.imread(old_file_path, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (224, 224))
            new_file_name = f"{category}_{image_index + 1}.jpg"
            new_file_path = os.path.join(category_output_path, new_file_name)
            cv2.imwrite(new_file_path, img)

# Example usage
# input_folder = "C:\\Users\\Alex\\Documents\\Alex\\master\\APDSV\\Proiect_2_0\\dataset\\garbage-dataset-edited\\batteries"   
input_folder = "C:\\Users\\Alex\\Documents\\Alex\\master\\APDSV\\Proiect_2_0\\dataset\\garbage-dataset-edited"
output_folder = "C:\\Users\\Alex\\Documents\\Alex\\master\\APDSV\\Proiect_2_0\\dataset\\dataset_final"  
rename_images_in_folder(input_folder, output_folder)
