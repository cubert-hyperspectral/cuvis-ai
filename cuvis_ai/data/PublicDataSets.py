import gdown
import os


class PublicDataSets:

    @classmethod
    def download_dataset(cls, dataset_name, *, download_path: str = ".", entries: list = None) -> bool:
        try:
            dset = cls._datasets[dataset_name]
        except KeyError:
            print(F"Dataset '{dataset_name}' not found.")
            return False

        if entries is None:
            to_download = dset[:]
        else:
            to_download = []
            for e in entries:
                try:
                    to_download.append(dset[e])
                except IndexError:
                    print(F"Entry {e} does not exist in dataset"
                          F" '{dataset_name}'")

        if len(to_download) == 0:
            print("Nothing to download.")
            return False

        if not os.path.exists(download_path):
            print(F"Directory '{download_path}' does not exist."
                  " It will be created.")
            os.makedirs(download_path)
        elif not os.path.isdir(download_path):
            print(
                F"Path '{download_path}' cannot be used. It points to an existing file.")
            return False

        items = []
        if isinstance(to_download[0], list):
            for entry in to_download:
                items.extend(entry)
            total_items = len(items)
            current_idx = 1
            print(F"Downloading {total_items} files from dataset"
                  F" '{dataset_name}'")
            for i in items:
                try:
                    gdown.download(id=i[0], output=os.path.join(download_path, i[1]), resume=True)
                except Exception as e:
                    print("Failed to fetch file:", i[1])
                    print("Error:", e)
        else:
            items = to_download
            total_items = len(items)
            current_idx = 1
            print(F"Downloading {total_items} folders from dataset"
                  F" '{dataset_name}'")
            for i in items:
                try:
                    gdown.download_folder(id=i[0], output=os.path.join(
                        download_path, i[1]))
                except Exception as e:
                    print("Failed to fetch folder:", i[1])
                    print("Error:", e)

        return True

    @classmethod
    def list_datasets(cls, verbose: bool = False):
        for name, data in cls._datasets.items():
            print(F"Dataset '{name}':")
            if isinstance(data[0], list):
                print(F" -> Contains {sum([len(d) for d in data])} items")
                if verbose:
                    print("  Listing items:")
                    count = 0
                    for d in data:
                        print("    Entry", count)
                        for i in d:
                            print("    ", i[1])
                        count += 1
            else:
                print(F" -> Contains {len(data)} datafolder(s)")
                if verbose:
                    for d in data:
                        print("    ", d[1])

    _datasets = {
        "GrowRipe_Samples": [
            [
                ("1cNDIG9nsHqGNb-wVsa8O3bVJ_bQ6Ls-b", "GrowRipe_Brix_8_2_A.cu3s"),
                ("1J0kZiy5Ht6qqETk_vpBPXBX_3dweUuHk", "GrowRipe_Brix_8_2_A.json"),
                ("1ocRJrRBrqtPpA_wMXYFp1GLOSvtJAOty", "GrowRipe_Brix_8_2_B.cu3s"),
                ("1bTPfFkoGLnI_FBXsEVnypN3SyFYOo0rD", "GrowRipe_Brix_8_2_B.json"),
            ],
            [
                ("1CRPZ7c5EBoAhPAKp6uIhohUTAHvaZYaY", "GrowRipe_Brix_7_7_A.cu3s"),
                ("1rDYbRa-dISAjfnSTeb6bO4P9379eOBiG", "GrowRipe_Brix_7_7_A.json"),
                ("1PjnE3pwGxZPJJFuUriYIF4txlCJ7M2c6", "GrowRipe_Brix_7_7_B.cu3s"),
                ("1eKKMxIKf9CzJj2f9pSNev-GCNfLTqK9w", "GrowRipe_Brix_7_7_B.json"),
            ],
            [
                ("1P_HKgaBUqOPObd9QMPouzWZCBmes9EKM", "GrowRipe_Brix_7_2_A.cu3s"),
                ("1tw1zlqn3EYDHssimANxOEUGjHRh2JQ_v", "GrowRipe_Brix_7_2_A.json"),
                ("15_8dOSnIC6GYWw_Bf9f12zO-pU80Xbi_", "GrowRipe_Brix_7_2_B.cu3s"),
                ("1y4ZeC1jbPLlZ0EyD1JZI_J866XT3K2Ou", "GrowRipe_Brix_7_2_B.json"),
            ],
            [
                ("1OS7lfA-D051pKOzLtPCZXBxRMFl_hTe_", "GrowRipe_Brix_6_8_A.cu3s"),
                ("1atPnU4ghyiSHRW6xjnEVk4rnb3Y4c6Ke", "GrowRipe_Brix_6_8_A.json"),
                ("115K-dZjrXiJq1ykig_eWMGKHNVd7Ps9I", "GrowRipe_Brix_6_8_B.cu3s"),
                ("1RhAuIpAKFt6OdDGsqdvJ2f20IEryRcN5", "GrowRipe_Brix_6_8_B.json"),
            ],
            [
                ("1B_mbHuVP6sAksB7_gHgbslkAblur8BPF", "GrowRipe_Brix_6_4_A.cu3s"),
                ("1wUk8RJiG_E5kNYXH-QMzwi4JNyvyL-Yr", "GrowRipe_Brix_6_4_A.json"),
                ("1TKEKho9GVmIVr2kUL_ANh87h9Fg1JYK6", "GrowRipe_Brix_6_4_B.cu3s"),
                ("1Iw3QGIR9NPeNn6xDLonWIpCp399MPsAv", "GrowRipe_Brix_6_4_B.json"),
            ],
            [
                ("1cPaz22IIagohswp2ewTUb4F1lG7y-W_H", "GrowRipe_Brix_5_8_A.cu3s"),
                ("1UtiJR3Hwp_TLn2qUJwXowYYw61okFtUO", "GrowRipe_Brix_5_8_A.json"),
                ("1ESRL4hy2hahUq2wh-3ElTg4L8WsLJC1Z", "GrowRipe_Brix_5_8_B.cu3s"),
                ("1paH74cP-dsymBEOgtvD-W8L30xNJ8YN-", "GrowRipe_Brix_5_8_B.json"),
            ],
            [
                ("1IreuvLnwibDtFBhfpp8b4icm-DQhVMYc", "GrowRipe_Brix_5_2_A.cu3s"),
                ("1O--6cKkLKPQWCb3WtasTPHjQFUVMTckk", "GrowRipe_Brix_5_2_A.json"),
                ("1lTQ6jC89geGERzBCdiCAlP6yz0YFMO7D", "GrowRipe_Brix_5_2_B.cu3s"),
                ("1w9ncos1BVLZzwNyaRPZ55WL6HC43kFbT", "GrowRipe_Brix_5_2_B.json"),
            ],
            [
                ("1ZkyvUoXGUDM_yOyr7-9XPg7nXQJQgz_K", "GrowRipe_Brix_4_6_A.cu3s"),
                ("1SRvdKj8phvb9H_yAidX_RvZFgk0thJOn", "GrowRipe_Brix_4_6_A.json"),
                ("1j6RM6pHMlSDY6ZpOi2GK0mw8RA_TK8zx", "GrowRipe_Brix_4_6_B.cu3s"),
                ("1nStSqvcsEYTitK6PXr-AtJBdgvu5BB9L", "GrowRipe_Brix_4_6_B.json"),
            ],
        ],
        "GrowRipe_FULL":
            [
                ("1MNitRfQJe9ZsHDRWGvT-7otW6yqmJyU_", "GrowRipe"),
        ],
        "Aquarium": [
            [
                ("1eFau2tUcke6hx5p1ESvK9X4ImovLkkrj", "Aquarium_Sample.cu3s"),
            ],
        ],
        "Lentils_Small": [
            [
            ("1vwj6IC0kTFDaflm_K-SOgf5AKmZKGrxr", "Lentils_000.cu3s"),
            ("1_nMjSjc6mowGfx4vaDbR6eAkS0I0nGye", "Lentils_000.info"),
            ("1MXcSPgHi0hl-VM37zDAXuspIs1-PNGff", "Lentils_000.json"),
            ]
        ],
    }
    #Alias
    _datasets["Lentils"] = _datasets["Lentils_Small"]
    
# if __name__ == "__main__":
#     PublicDataSets.list_datasets(verbose=True)
#     base_path = "../data/Lentils_SMALL"
#     data_down = PublicDataSets()
#     data_down.list_datasets(verbose=True)
#     #data_down.download_dataset("Lentils_SMALL", download_path=base_path)
#     data_down.download_dataset("Lentils_2_0_XMR", download_path=base_path)
