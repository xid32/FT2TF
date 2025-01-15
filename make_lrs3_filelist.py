import os

def list_txt_files(directory, output_file):
    with open(output_file, 'w', encoding='utf-8') as out_file:
        # Walk through the specified directory and its subdirectories
        for foldername, subfolders, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith('.txt'):
                    # Get the full file path
                    full_path = os.path.join(foldername, filename)
                    # Remove the .txt extension from the filename
                    path_without_txt = full_path[:-4]
                    path_without_txt = path_without_txt.replace("/jumbo/jinlab/datasets/lrs3/", "")
                    # Write the path without .txt extension to the output file
                    out_file.write(path_without_txt + '\n')

if __name__ == "__main__":
    if not os.path.exists("/jumbo/jinlab/XD/face_expression/lrs3_filelist"):
        os.makedirs("/jumbo/jinlab/XD/face_expression/lrs3_filelist")
    directory = "/jumbo/jinlab/datasets/lrs3/test"
    output_file = "/jumbo/jinlab/XD/face_expression/lrs3_filelist/test.txt"
    list_txt_files(directory, output_file)

    directory = "/jumbo/jinlab/datasets/lrs3/trainval"
    output_file = "/jumbo/jinlab/XD/face_expression/lrs3_filelist/train.txt"
    list_txt_files(directory, output_file)
