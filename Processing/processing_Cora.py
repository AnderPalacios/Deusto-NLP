import os
import pandas as pd
import re
from collections import defaultdict


class ProcessCora():
    def __init__(self):
        self.N_EDGES_CORA = 5429
        self.N_NODES_CORA = 2708
        self.ALL_EXTRACTIONS = "./cora_extracted/cora_orig/mccallum/cora/extractions/"    
        self.ORIGINAL_CORA = "./cora_extracted/cora_orig/cora.cites"
        self.BOW_CORA = "./cora_extracted/cora_orig/cora.content"
        self.FILENAMES_PATH = "./cora_extracted/cora_orig/mccallum/cora/papers"
        self.CLASS_PATH = "./cora_extracted/cora_orig/mccallum/cora/classifications"
        self.ALL_ITEMS = os.listdir(self.ALL_EXTRACTIONS)  
        self.DATA_FILES = [f for f in self.ALL_ITEMS if os.path.isfile(os.path.join(self.ALL_EXTRACTIONS, f))]



    def get_info_ids(self, path):
        paper_ids = set()
        with open(path, "r") as f:
            lines = f.readlines()
            for _, line in enumerate(lines):
                line = line.strip()
                cited_id, citing_id = map(int, line.split('\t'))
                paper_ids.add(cited_id)
                paper_ids.add(citing_id)
        print(f"Number of unique IDs equal to number of Nodes in Cora: {self.N_NODES_CORA == len(paper_ids)}")
        print(f"Number of citations equal to number of Edges in Cora: {self.N_EDGES_CORA == len(lines)}")
        return paper_ids


    def get_filenames(self, filename_path):
        data_dict = defaultdict(list)

        with open(filename_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    paper_id = int(parts[0])
                    url = parts[1]
                    data_dict[paper_id].append(url)

        # Convert to a list of dictionaries
        data_list = [{paper_id: urls} for paper_id, urls in data_dict.items()]
        return data_list



    def get_original_Cora(self, orig_ids, filenames):
        filtered_list = [d for d in filenames if list(d.keys())[0] in orig_ids]
        print(f"Equal number of ids and nodes -> {len(filtered_list) == self.N_NODES_CORA}")
        print(f"Same ids as paper Ids -> {sorted(orig_ids) == sorted(list(d.keys())[0] for d in filtered_list)}")
        return filtered_list



    def get_startings(self, ids_filenames):
        protocols = set()

        for entry in ids_filenames:
            for urls in entry.values():
                for url in urls:
                    protocol = url.split("##")[0]  # get part before ##
                    protocols.add(protocol)

        return protocols



    def update_protocol(self, dataset):
        fixed_data = []

        for entry in dataset:
            new_entry = {}
            for pid, urls in entry.items():
                new_urls = []
                for url in urls:
                    # Replace all colons with underscores
                    new_url = url.replace(":", "_")
                    new_urls.append(new_url)
                new_entry[pid] = new_urls
            fixed_data.append(new_entry)

        return fixed_data



    def check_missing_data(self, dataset):
        found_count = 0
        filtered_dataset = []

        for entry in dataset:
            keep_entry = {}
            for paper_id, urls in entry.items():
                # Keep the entry if at least one URL is found in DATA_FILES
                if any(url in self.DATA_FILES for url in urls):
                    found_count += 1
                    keep_entry[paper_id] = urls
            if keep_entry:  # Only append if there is at least one valid paper
                filtered_dataset.append(keep_entry)

        missing_count = sum(len(entry) for entry in dataset) - found_count

        print(f"Papers with info: {found_count}")
        print(f"Papers missing info: {missing_count}")
        print(f"Total papers: {found_count + missing_count}")

        return filtered_dataset



    def extract_paper_info2(self, dataset, data_folder):
        """
        Extracts Title, Author, and Abstract from paper files.

        Only includes papers in the returned dictionary if both Title and Abstract are found.

        Args:
            dataset (list of dicts): [{paper_id: [url1, url2, ...]}, ...]
            data_folder (str): Folder where the downloaded paper info files are stored

        Returns:
            dict: {paper_id: {"Title": ..., "Author": ..., "Abstract": ...}, ...}
        """
        paper_info = {}
        num_missing_abs = 0

        for entry in dataset:
            for paper_id, urls in entry.items():
                found_info = False  
                title = None
                author = None
                abstract = None

                for url in urls:
                    # Convert URL to a local filename (full URL)
                    filename = os.path.join(data_folder, url)
                    
                    if os.path.exists(filename):
                        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        abstract_found = None

                        for line in content.splitlines():
                            if line.startswith("Title:") and title is None:
                                title = line.replace("Title:", "").strip()
                            elif line.startswith("Author:") and author is None:
                                author = line.replace("Author:", "").strip()
                            elif line.startswith("Abstract:") and abstract is None:
                                abstract = line.replace("Abstract:", "").strip()
                            elif line.startswith("Abstract-found:") and abstract_found is None:
                                abstract_found = line.replace("Abstract-found:", "").strip()
                        
                        # Stop checking URLs if abstract is found
                        if abstract_found == "1" and abstract is not None:
                            found_info = True
                            break
                    else:
                        print(f"File not found: {filename}")

                if not found_info:
                    num_missing_abs += 1

                # Only store paper if both Title and Abstract are not None
                if title is not None and abstract is not None:
                    paper_info[paper_id] = {
                        "Title": title,
                        "Author": author,
                        "Abstract": abstract
                    }

        return paper_info



    def list_to_df(self, dataset):
        # Flatten the list of dicts into a list of tuples (paper_id, filenames)
        flat_data = [(paper_id, urls) for entry in dataset for paper_id, urls in entry.items()]

        df = pd.DataFrame(flat_data, columns=["paper_id", "filenames"])
        return df



    def dict_to_df(self, dict_info):
        df = pd.DataFrame.from_dict(dict_info, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'paper_id'}, inplace=True)
        return df



    def merge_dfs(self, df1, df2):
        merged_df = pd.merge(df1, df2, on="paper_id", how="inner")
        return merged_df


    def check_NA(self, df):
        num_missing_AB = df["Abstract"].isna().sum()
        num_missing_TI = df["Title"].isna().sum()
        print(f"Number of missing abstract: {num_missing_AB}")
        print(f"Number of missing abstract: {num_missing_TI}")


    def map_urls_topic(self, file_path):
        url_to_topic = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    print(idx)
                url, topic = line.split("\t")
                url_to_topic[url] = topic
        return url_to_topic


    def normalize_url(self, url):
        return re.sub(r'^(http|https|ftp|file)_', r'\1:', url)


    def get_paper_topic(self, mapping_urls_topics, url_list, paper_id):
        for url in url_list:
            # Normalize URL
            norm_url = self.normalize_url(url)
            if norm_url in mapping_urls_topics:
                # Return only the last part of the topic
                return mapping_urls_topics[norm_url].rstrip("/").split("/")[-1]
        # If no match found
        return None


    def normalize_url2(self, url):
        return url.replace("_", ":")


    def get_paper_topic2(self, mapping_urls_topics, url_list, paper_id):
        for url in url_list:
            norm_url = self.normalize_url2(url)
            if norm_url in mapping_urls_topics:
                return mapping_urls_topics[norm_url].rstrip("/").split("/")[-1]
        return None


    def manual_assignments(self, df):
        missing_topic_ids = df[df["topic"].isna()]["paper_id"].tolist()
        if sorted(missing_topic_ids) != sorted([46500, 85452, 1111733, 1116530, 1125258]):
            raise Exception("These are not the papers with missing topics")

        df.loc[df["paper_id"] == 46500, "topic"] = "Theory"
        df.loc[df["paper_id"] == 85452, "topic"] = "Theory"
        df.loc[df["paper_id"] == 1111733, "topic"] = "Neural_Networks"
        df.loc[df["paper_id"] == 1116530, "topic"] = "Probabilistic_Methods"
        df.loc[df["paper_id"] == 1125258, "topic"] = "Neural_Networks"
        return df



    def get_labels(self, path, df):
        mappings = self.map_urls_topic(path)  
        df["topic"] = df.apply(lambda row: self.get_paper_topic(mappings, row["filenames"], row["paper_id"]), axis=1)
        # The rest:
        mask = df["topic"].isna()
        df.loc[mask, "topic"] = df[mask].apply(lambda row: self.get_paper_topic2(mappings, row["filenames"], row["paper_id"]), axis=1)
        df = self.manual_assignments(df)
        return df



    def get_Bow(self, path):
        data = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue  # skip malformed lines
                paper_id = int(parts[0])
                topic2 = parts[-1]  # last element is the topic
                bow = list(map(int, parts[1:-1]))  # all elements between paper_id and topic are BoW
                data.append({
                    "paper_id": paper_id,
                    "BoW": bow,
                    "topic2": topic2
                })

        df = pd.DataFrame(data)
        return df



    def merge_info_bow(self, df1, df2):
        df = pd.merge(df1, df2, on="paper_id", how="inner")
        df.drop(columns=["topic"], inplace=True)
        return df
    
    def to_csv(self, dataframe):
        dataframe.to_csv("Cora_dataset.csv", index=False, encoding="utf-8")