from collections import Counter
import json
def load_data(data_path):
    with open(data_path, 'r') as file:
        data_json = json.load(file)

    text_lengths = [len(review["text"]) for review in data_json]
    star_values = [review["stars"] for review in data_json]

    min_length = min(text_lengths)
    max_length = max(text_lengths)
    avg_length = sum(text_lengths) / len(text_lengths)

    avg_star = sum(star_values) / len(star_values)

    star_counts = Counter(star_values)

    print(f"Total count:", len(data_json))
    print(f"Minimum text length: {min_length}")
    print(f"Maximum text length: {max_length}")
    print(f"Average text length: {avg_length:.2f}")
    print(f"Average star rating: {avg_star:.2f}")
    print("Star rating counts:")
    for star, count in sorted(star_counts.items()):
        print(f"  {star} star(s): {count}")

print("\n====TRAINING====")
load_data('./training.json')
print("\n====TESTING====")
load_data('./test.json')
print("\n====VALIDATION====")
load_data('./validation.json')