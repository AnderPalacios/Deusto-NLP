import pandas as pd


class ProcessPubMed():
    def __init__(self):
        self.NUM_NODES_PUBMED = 19717
        self.FAILED_PMID = 17874530
        self.KEEP_KEYS = ["PMID", "TI", "AB", "MH", "FAU"]
        self.PUBMED_JSON = "./PubMed_orig/PubMed_orig/pubmed.json"
        self.FILE_PATH = "./PubMed_orig/PubMed_orig/data/Pubmed-Diabetes.NODE.paper.tab"



    # Look for missing authors, titles or abstracts
    def get_data_info(self, dataset):
        missing_identifiers = [1 for p in dataset if not p.get("PMID")]
        missing_titles = [p.get("PMID") for p in dataset if not p.get("TI")]
        missing_abstracts = [p.get('PMID') for p in dataset if not p.get("AB")]
        missing_med_headdings = [p.get('PMID') for p in dataset if not p.get("MH")]
        missing_authors = [p.get('PMID') for p in dataset if not p.get("FAU")]

        print(f"Missing identifiers: {len(missing_identifiers)}")
        print(f"Missing titles: {len(missing_titles)}")
        print(f"Missing abstracts: {len(missing_abstracts)}")
        print(f"Missing medical headers: {len(missing_med_headdings)}")
        print(f"Missing authors: {len(missing_authors)}")



    def get_valid_papers(self, dataset):
        data = [p for p in dataset if p.get("TI") and p.get("AB")]
        return data


    # Function to get useful features
    def get_features(self, dataset):
        return [{k: p[k] for k in self.KEEP_KEYS if k in p} for p in dataset]



    def to_pandas(self, dataset):
        return pd.DataFrame(dataset)




    def flatten_lists(self, dataset):
        dataset['MH'] = dataset['MH'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
        dataset['FAU'] = dataset['FAU'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
        return dataset



    def get_missing(self, dataframe):
        return dataframe.isna().sum()



    def parse_columns(self, dataframe):
        df = dataframe.astype({"PMID":"int64", "TI":str, "AB":str})
        return df


    def compare_files(self, file_path, original_df):
        df2 = pd.read_csv(file_path, sep='\t', skiprows=1, low_memory=False)  

        df2 = df2[df2.iloc[:, 0] != self.FAILED_PMID]
        paper_ids = df2.iloc[:, 0].tolist()

        if len(paper_ids) != len(original_df):
            raise Exception("Different number of IDs or different IDs")
        else:
            print("Same IDs in both datasets")



    # Function to parse each line
    def parse_tfid(self, line):
        parts = line.strip().split('\t')
        pmid = int(parts[0])
        data = {"PMID": pmid, "label": None, "tfidf_words": {}, "summary_words": []}

        for part in parts[1:]:
            if part.startswith("label="):
                data["label"] = int(part.split("=")[1])
            elif part.startswith("w-") and "=" in part:
                # Extract TF-IDF word and value
                word, val = part.split("=")
                word = word.replace("w-", "")
                data["tfidf_words"][word] = float(val)
            elif part.startswith("summary="):
                words = part.replace("summary=", "").split(",")
                words_clean = [w.strip().replace("w-", "") for w in words]
                data["summary_words"] = words_clean

        return data


    def create_df(self, file_path):
        entries = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Skip the first 2 header lines
        for line in lines[2:]:
            if line.strip():  # skip blank lines
                entries.append(self.parse_tfid(line))

        df = pd.DataFrame(entries)
        df = df[df.iloc[:, 0] != self.FAILED_PMID]
        return df



    def merge_datasets(self, original_df, tfidf_df):
        df = pd.merge(original_df, tfidf_df, on="PMID")
        df = df.rename(columns={"TI":"Title", "AB":"Abstract","MH":"Key_words", "FAU":"Authors", "tfidf_words": "TFIDF"})
        return df
    
    def to_csv(self, dataframe):
        dataframe.to_csv("PubMed_dataset.csv", index=False, encoding="utf-8")