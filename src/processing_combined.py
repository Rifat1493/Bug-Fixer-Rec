import pandas as pd
from sklearn.model_selection import train_test_split

def generate_combined_data():
    """
    Reads bug report and metadata CSV files, merges them on bug ID, and
    saves the combined data to data/bug_repo_combined_data.csv.
    """

    df_raw = pd.read_csv('data/all_data/CSV_100k_filtered_bug_reports.csv', usecols=['bug_id','text', 'resolution'],low_memory=False, encoding='ISO-8859-1')
    df_raw.dropna(inplace=True)
    df_meta_data = pd.read_csv('data/all_data/Bug_meta_data.csv',usecols=['id','summary','assigned_to'],low_memory=False, encoding='ISO-8859-1')
    df_meta_data.dropna(inplace=True)
    df_raw['bug_id'] = df_raw['bug_id'].astype(int)
    df_combined = df_raw.merge(df_meta_data, left_on='bug_id', right_on='id')
    df_combined.to_csv('data/bug_repo_combined_data.csv', index=False)

def generate_train_test_split():
    """
    Generates stratified train and test CSV splits from a sampled subset of
    bug report data, focusing on the top 250 developers by assignment
    frequency.
    """

    df = pd.read_csv('data/bug_repo_combined_data.csv')
    # df = pd.read_excel("data/all_data/Good_Bug_reports_with_S2r_AR_ER.xlsx", usecols=["text", "assigned_to"])

    # df = df.sample(frac=0.05, random_state=42)
    df_developers_profile = (
        df.groupby('assigned_to')
        .size()
        .reset_index(name='number_of_instances')
        .sort_values('number_of_instances', ascending=False)
        .head(100)
        .reset_index(drop=True)
    )
    top_250_developers = df_developers_profile['assigned_to'].head(100).tolist()
    df_top_250 = df[df['assigned_to'].isin(top_250_developers)]
    df_train, df_test = train_test_split(
        df_top_250,
        test_size=0.20,
        random_state=42,
        stratify=df_top_250['assigned_to']
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_train.to_csv('data/train.csv', index=False)
    df_test.to_csv('data/test.csv', index=False)

    print(f"Total instances (Top 250 Developers): {len(df_top_250)}")
    print(f"Training set size: {len(df_train)} ({len(df_train)/len(df_top_250)*100:.1f}%)")
    print(f"Test set size:     {len(df_test)} ({len(df_test)/len(df_top_250)*100:.1f}%)")
    print(f"\nUnique developers in Train: {df_train['assigned_to'].nunique()}")
    print(f"Unique developers in Test:  {df_test['assigned_to'].nunique()}")

if __name__ == "__main__":
    # generate_combined_data()
    generate_train_test_split()