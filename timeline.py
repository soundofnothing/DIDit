from typing import List, Tuple, Generator
import enum
import requests
import feedparser
from datetime import datetime

class AutoName(enum.EnumMeta):
    def _generate_next_value_(name, start, count, last_values):
        return name

class RSSFeed(enum.Enum, metaclass=AutoName):
    ACCOUNT1 = enum.auto()
    ACCOUNT2 = enum.auto()
    ACCOUNT3 = enum.auto()
    # Add more members as needed for other accounts

def get_timeline_data(rss_feed: RSSFeed) -> List[Tuple[datetime, str]]:
    # Get the RSS feed URL based on the selected enum member
    rss_feed_url = get_rss_feed_url(rss_feed)

    # Fetch the RSS feed from the provided URL
    feed = feedparser.parse(rss_feed_url)

    # Extract the timeline entries and their corresponding dates
    timeline_entries = []
    for entry in feed.entries:
        date = parse_timestamp(entry.published)
        text = parse_text(entry.title)
        timeline_entries.append((date, text))

    return timeline_entries


def parse_timestamp(timestamp: str) -> datetime:
    # Parse the timestamp using datetime
    date = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    return date


def parse_text(text: str) -> str:
    # Return the parsed text without any modifications
    return text


def generate_timeline_timeseries(rss_feed: RSSFeed) -> Generator[Tuple[datetime, str], None, None]:
    # Get the RSS feed URL based on the selected enum member
    rss_feed_url = get_rss_feed_url(rss_feed)

    # Fetch the RSS feed and yield each entry as a timeseries
    feed = feedparser.parse(rss_feed_url)
    for entry in feed.entries:
        date = parse_timestamp(entry.published)
        text = parse_text(entry.title)
        yield (date, text)


def get_rss_feed_url(rss_feed: RSSFeed) -> str:
    # Replace this with the actual logic to get the RSS feed URL for the selected account
    url_map = {
        RSSFeed.ACCOUNT1: "https://example.com/api/account1/rss",
        RSSFeed.ACCOUNT2: "https://example.com/api/account2/rss",
        RSSFeed.ACCOUNT3: "https://example.com/api/account3/rss",
        # Add more mappings as needed
    }
    return url_map.get(rss_feed, "")  # Return an empty string if the mapping is not found


if __name__ == "__main__":
    # Example usage
    rss_feed = RSSFeed.ACCOUNT1
    timeline_data = get_timeline_data(rss_feed)
    print(timeline_data)

    timeseries = generate_timeline_timeseries(rss_feed)
    for entry in timeseries:
        print(entry)
