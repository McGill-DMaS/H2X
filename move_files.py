import os
import shutil

# Define the source and destination base directories
source_dir = 'E:\\saqib_work1\\data\\miles\\benign_0'
dest_base_dir = 'E:\\saqib_work1\\data\\miles\\to_excute_benign_5\\benign_au_5'

# Number of files to move in each batch
batch_size = 20
total_files_to_move = 200


def move_files_in_batches(source, dest_base, batch_size, total_files):
    # Get a list of all files in the source directory
    all_files = os.listdir(source)
    all_files = [f for f in all_files if os.path.isfile(os.path.join(source, f))]

    if len(all_files) < total_files:
        raise ValueError(
            f"Source directory does not contain enough files. Found {len(all_files)} files, but need {total_files}.")

    for i in range(0, total_files, batch_size):
        batch_files = all_files[i:i + batch_size]
        dest_dir = f"{dest_base}{i // batch_size + 1}"

        # Create the destination directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)

        # Move each file in the batch
        for file in batch_files:
            shutil.move(os.path.join(source, file), os.path.join(dest_dir, file))
        print(f"Moved batch {i // batch_size + 1} to {dest_dir}")


# Run the function to move files
move_files_in_batches(source_dir, dest_base_dir, batch_size, total_files_to_move)
