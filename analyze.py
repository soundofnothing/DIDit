import requests
from typing import List


def get_timeline(screen_name: str, url: str) -> List[str]:
    try:
        response = requests.get(url)
        response.raise_for_status()
        timeline_data = response.json()
        timeline_posts = [post['text'] for post in timeline_data]
        return timeline_posts
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return []


def analyze_timeline(screen_name: str, url: str):
    timeline_posts = get_timeline(screen_name, url)
    if not timeline_posts:
        print("Unable to retrieve the timeline data.")
        return
    
    # Analyze the timeline posts
    # ...

    # Example: Print the timeline posts
    for post in timeline_posts:
        print(post)


# Example usage
screen_name = "example_user"
url = "https://api.example.com/timeline"
analyze_timeline(screen_name, url)
