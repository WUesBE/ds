import re
import string
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')


class DataCleaner:
    def __init__(self, df):
        self.df = df
        self.column_scope = ['comment_count', 'dislike_count', 'like_count', 'view_count', 'licensed_content',
                             'duration_sec', 'video_category_label', 'video_title', 'video_description', 'published_at',
                             'definition', 'duration']
    def clean_desc(self, desc):
        desc = desc.split('#BBC')[0]
        desc = re.split(r' https://bbc\.in\w*', desc)[-1]
        return desc
    def clean_licensed(self, lin):
        try:
            int(lin)
            return True
        except ValueError:
            if type(lin) is float:
                return False

    def parse_duration(self, duration_str):
        duration_str = duration_str[2:]
        total_seconds = 0
        if 'H' in duration_str:
            hours, duration_str = duration_str.split('H')
            total_seconds += int(hours) * 3600
        if 'M' in duration_str:
            minutes, duration_str = duration_str.split('M')
            total_seconds += int(minutes) * 60
        if 'S' in duration_str:
            seconds = duration_str.rstrip('S')
            total_seconds += int(seconds)

        return total_seconds
    def clean(self):
        self.df = self.df[self.column_scope]
        self.df['video_description_cleaned'] = self.df['video_description'].apply(cleaner.clean_desc)
        self.df['licensed_content'] = self.df['licensed_content'].apply(cleaner.clean_licensed)
        self.df['published_at'] = self.df['published_at'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))
        self.df['duration'] = self.df['duration'].apply(self.parse_duration)
        print(self.df.describe())

        return self.df


class BBC:
    def __init__(self, df):
        self.df = df
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stop_words = set(stopwords.words('english'))
        self.stop_words = self.stop_words.union({'bbc', 'two', 'one', 'three',
                              'part', 'series', 'episode', 'preview', 'show'})

    def data_types(self):
        for column in list(self.df.columns):
            print(f'column name: {column},\n column values types: {self.df[column].apply(type).unique()},\n'
                  f' representative value: {self.df[column].unique()[1]}\n###################\n')

    def date_check(self):
        newest_date, oldest_date = self.df['published_at'].max().year, self.df['published_at'].min().year
        print(f'newest article (YYYY): {newest_date},'
              f'oldest article (YYYY): {oldest_date}')
        self.df['published_parsed'] = self.df['published_at'].dt.year

    def top_categories(self):
        top_occurrences = self.df['video_category_label'].value_counts().head(5)
        colors = ['#004c6d', '#00587a', '#00668e', '#0074a2',  '#0083b6',
                  '#0091c9', '#00a0dd', '#00aee1', '#00bcf4', '#00cbff']

        plt.figure(figsize=(8, 8))
        patches, texts, autotexts = plt.pie(top_occurrences, labels=top_occurrences.index, autopct='%1.1f%%', colors=colors)
        for autotext in autotexts:
            autotext.set_color('white')
        plt.title('Top 5 Occurrences in DataFrame Column')
        plt.show()

    def clean_title(self):
        def _clean_title(title):

            title = title.translate(str.maketrans('', '', string.punctuation))
            title = re.sub(r'\d+', '', title)
            tokens = word_tokenize(title)
            tokens = [word.lower() for word in tokens if word.lower() not in self.stop_words]


            return tokens
        #leaving in tokens form as it will come in handy in the next task
        self.df['video_title_clean'] = self.df['video_title'].apply(_clean_title)

    def top_title_keywords(self):
        _df = self.df
        grouped = _df.groupby('published_parsed')['video_title_clean'].sum()
        keyword_counts = grouped.apply(Counter)
        top_keywords_by_year = keyword_counts.apply(lambda x: [word for word, _ in x.most_common(5)])
        print(top_keywords_by_year)
        top_keywords_by_year = keyword_counts.apply(lambda x: [word for word, _ in x.most_common(2)])
        keyword_counts = top_keywords_by_year.explode().groupby(level=0).value_counts().unstack(fill_value=0)
        keyword_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title('Top 2 Keywords in Titles Each Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Occurrences')
        plt.legend(title='Keywords')
        plt.xticks(rotation=45)
        plt.show()
        del _df


    def engagement_rate(self):
        self.df['engagement_rate'] = ((self.df['like_count'] + self.df['comment_count'] + self.df['dislike_count']) / self.df['view_count']) * 100
        self.df['engagement_rate'] = self.df['engagement_rate'].round(1)

    def title_len(self):
        self.df['title_len'] = self.df['video_title_clean'].apply(lambda x: sum(len(word) for word in x))

    def dichotomized_engagement(self):
        median_rate = self.df['engagement_rate'].median()
        self.df['dichotomized_engagement'] = (self.df['engagement_rate'] >= median_rate).astype(int)

    def encode2numeric(self, *args):
        for column in args:
            self.df[column], _ = pd.factorize(self.df[column])

    def visualise_correlations(self):
        correlation_matrix = self.df[
            ['definition', 'duration', 'dichotomized_engagement', 'published_parsed', 'engagement_rate', 'title_len',
             'video_category_label']].corr()

        plt.figure(figsize=(10, 8))
        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
        plt.colorbar()
        plt.title('Correlation Matrix of BBC YouTube Videos Metadata')
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        plt.show()

        plt.figure(figsize=(12, 8))
        for i, col in enumerate(['duration', 'engagement_rate', 'title_len']):
            plt.subplot(2, 2, i + 1)
            plt.scatter(self.df[col], self.df['dichotomized_engagement'], alpha=0.5)
            plt.xlabel(col)
            plt.ylabel('Dichotomized Score')
            plt.title(f'Scatter plot: Dichotomized Score vs {col}')
        plt.tight_layout()
        plt.show()

    def run(self):
        self.data_types()
        self.date_check()
        self.df.drop(columns=['published_at', 'licensed_content', 'duration'], inplace=True)
        # IMO 'licensed_content' and 'duration_sec' may bring valuable correlations while analyzing the dataset
        # ... but im dropping them since thats the task (:
        self.top_categories()
        self.clean_title()
        self.top_title_keywords()
        self.engagement_rate()
        self.title_len()
        self.dichotomized_engagement()
        self.encode2numeric('video_category_label', 'definition')
        self.visualise_correlations()


if __name__ == '__main__':
    bbc_csv = pd.read_csv(r'C:\Users\mikol_xrn4\PycharmProjects\Krzyzowki\bbc.csv')
    cleaner = DataCleaner(bbc_csv)
    cleaned_df = cleaner.clean()
    bbc = BBC(cleaned_df)
    bbc.run()
