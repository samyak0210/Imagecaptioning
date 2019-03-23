import csv

def load_captions():
    caption_dict = {}
    with open("results.csv") as file_obj:
        reader = csv.DictReader(file_obj, delimiter='|')
        for line in reader:
            if line['image_name'] in caption_dict:
                if line[' comment'] is None:
                    continue
                if len(line[' comment']) > len(caption_dict[line['image_name']]):
                    caption_dict[line['image_name']] = line[' comment']
            else:
                caption_dict[line['image_name']] = line[' comment']
    return caption_dict

