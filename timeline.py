from typing import List, Tuple, Generator, Union
import enum
import requests
import feedparser
from datetime import datetime
import csv


class AutoName(enum.EnumMeta):
    def _generate_next_value_(name, start, count, last_values):
        return name


class DataSource(enum.Enum, metaclass=AutoName):
    RSS_FEED = enum.auto()
    CSV_FILE = enum.auto()


class RSSFeed(enum.Enum, metaclass=AutoName):
    ACCOUNT1 = enum.auto()
    ACCOUNT2 = enum.auto()
    ACCOUNT3 = enum.auto()
    # Add more members as needed for other accounts


class DataImporter:
    def __init__(self, data_source: DataSource, source_url: str = None):
        self.data_source = data_source
        self.source_url = source_url

    def import_data(self) -> List[Tuple[datetime, str]]:
        if self.data_source == DataSource.RSS_FEED:
            return self._import_from_rss_feed()
        elif self.data_source == DataSource.CSV_FILE:
            return self._import_from_csv_file()
        else:
            raise ValueError("Invalid data source")

    def _import_from_rss_feed(self) -> List[Tuple[datetime, str]]:
        rss_feed_url = self._get_rss_feed_url()

        feed = feedparser.parse(rss_feed_url)

        timeline_entries = []
        for entry in feed.entries:
            date = self._parse_timestamp(entry.published)
            text = self._parse_text(entry.title)
            timeline_entries.append((date, text))

        return timeline_entries

    def _import_from_csv_file(self) -> List[Tuple[datetime, str]]:
        with open(self.source_url, 'r') as file:
            reader = csv.DictReader(file)
            timeline_entries = []
            for row in reader:
                date = self._parse_timestamp(row['date'])
                text = self._parse_text(row['text'])
                timeline_entries.append((date, text))
            return timeline_entries

    def _get_rss_feed_url(self) -> str:
        url_map = {
            RSSFeed.ACCOUNT1: "https://example.com/api/account1/rss",
            RSSFeed.ACCOUNT2: "https://example.com/api/account2/rss",
            RSSFeed.ACCOUNT3: "https://example.com/api/account3/rss",
            # Add more mappings as needed
        }
        return url_map.get(self.source_url, "")

    @staticmethod
    def _parse_timestamp(timestamp: str) -> datetime:
        date = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
        return date

    @staticmethod
    def _parse_text(text: str) -> str:
        return text


def generate_timeline_timeseries(data_source: DataSource, source_url: str = None) -> Generator[
    Tuple[datetime, str], None, None]:
    importer = DataImporter(data_source, source_url)
    timeline_data = importer.import_data()

    for entry in timeline_data:
        yield entry


if __name__ == "__main__":
    # Example usage
    rss_feed = RSSFeed.ACCOUNT1
    timeline_data = generate_timeline_timeseries(DataSource.RSS_FEED, rss_feed.value)
    print(timeline_data)

    csv_file = "path/to/data.csv"
    timeline_data = generate_timeline_timeseries(DataSource.CSV_FILE, csv_file)
    print(timeline_data)
