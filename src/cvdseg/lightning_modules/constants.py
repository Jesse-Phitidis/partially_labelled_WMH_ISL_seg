mutually_exclusive_labels = ["WMH", "ISLtotal", "ICH", "TUMOUR"]
index_to_class = {1: "WMH", 2: "ISLtotal", 3: "LACUNE", 4: "ICH", 5: "TUMOUR", 6: "PVS", 7: "CMB"}
class_to_index = {v: k for k, v in index_to_class.items()}