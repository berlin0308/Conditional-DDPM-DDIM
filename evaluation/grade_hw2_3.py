import json
import argparse
import os
from clip_image_score import calculate_clip_image_scores_folder
from clip_text_score import calculate_clip_text_scores_folder

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process some JSON data.')
    parser.add_argument('--json_path', type=str, help='Path to the JSON file')
    parser.add_argument('--input_dir', type=str, default='.', help='Directory of the hw2_data/textual_inversion')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory of saved output')
    
    # Parse the arguments
    args = parser.parse_args()
    json_path = args.json_path
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Assuming the JSON is saved in a file named 'data.json'
    with open(json_path, 'r') as f:
        data = json.load(f)

    pass_count = 0
    count = 0
    print("\n===============================================start evaluation================================================\n")

    # Iterate through the data and print the prompt_4_clip_eval
    for key, value in data.items():
        input_folder_path = os.path.join(input_dir, key)
        output_folder_path = os.path.join(output_dir, key)

        src = value['src_image']

        for idx, clip_eval in enumerate(value['prompt_4_clip_eval']):
            prompt_output_folder_path = os.path.join(output_folder_path, str(idx))
            print(f"Image source: \"{src}\", text prompt: {clip_eval}")

            image_scores = calculate_clip_image_scores_folder(prompt_output_folder_path, input_folder_path)
            text_scores = calculate_clip_text_scores_folder(prompt_output_folder_path, clip_eval)
            
            # total_score = image_scores + 2.5 * text_scores
            total_score = [i + 2.5 * j for i, j in zip(image_scores, text_scores)]
            sorted_indices = sorted(range(len(total_score)), key=lambda k: total_score[k])[-5:]
            image_score = sum(image_scores[i] for i in sorted_indices) / len(sorted_indices)
            text_scores = sum(text_scores[i] for i in sorted_indices) / len(sorted_indices)


            print(f"CLIP Image Score: {image_score:.2f}")
            print(f"CLIP Text Score: {text_scores:.2f}")

            baseline = value['baseline'][idx]
            if image_score >= baseline[0] and text_scores >= baseline[1]:
                print("\n=====================================================PASS======================================================\n")

                pass_count += 1
            else:
                print(f"\n=========================================Fail! Baseline is {baseline[0]:.2f}/{baseline[1]:.2f}=========================================\n")
            
            count += 1

    print(f"You have passed {pass_count}/{count} cases")


    