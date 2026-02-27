import os
import csv


def clean_attack(x: str) -> str:
    x = (x or "").strip()
    if x.endswith("."):
        x = x[:-1]
    return x


def get_or_create_id(d: dict, key: str) -> int:
    if key not in d:
        d[key] = len(d)
    return d[key]


def rawfile2Multiclass(
    rootfolder,
    choice_dataset,   # 1=KDDCup99, 2=NSL-KDD
    choice_split,     # KDD: 1=labeled(corrected) 2=unlabeled ; NSL: 1=train 2=test
    protocol_map=None,
    service_map=None,
    flag_map=None,
    label_map=None,
):
    if protocol_map is None:
        protocol_map = {}
    if service_map is None:
        service_map = {}
    if flag_map is None:
        flag_map = {}
    if label_map is None:
        label_map = {}

    attack2family = {
        "back": "dos",
        "buffer_overflow": "u2r",
        "ftp_write": "r2l",
        "guess_passwd": "r2l",
        "imap": "r2l",
        "ipsweep": "probe",
        "land": "dos",
        "loadmodule": "u2r",
        "multihop": "r2l",
        "neptune": "dos",
        "nmap": "probe",
        "perl": "u2r",
        "phf": "r2l",
        "pod": "dos",
        "portsweep": "probe",
        "rootkit": "u2r",
        "satan": "probe",
        "smurf": "dos",
        "spy": "r2l",
        "teardrop": "dos",
        "warezclient": "r2l",
        "warezmaster": "r2l",
        "normal": "normal",
        "unlabeled": "unknown",
    }

    files = os.listdir(rootfolder)
    dataset_files = []

    # ---- KDDCup99 ----
    if choice_dataset == 1:
        for fname in files:
            fullpath = os.path.join(rootfolder, fname)
            if not (fname.startswith("kddcup") and os.path.isfile(fullpath)):
                continue

            if choice_split == 1:
                if ("unlabeled" not in fname) and ("corrected" in fname):
                    dataset_files.append(fname)
            else:
                if "unlabeled" in fname:
                    dataset_files.append(fname)

    # ---- NSL-KDD ----
    else:
        for fname in files:
            fullpath = os.path.join(rootfolder, fname)
            if not os.path.isfile(fullpath):
                continue

            if choice_split == 1 and fname.startswith("KDDTrain"):
                dataset_files.append(fname)
            elif choice_split == 2 and fname.startswith("KDDTest"):
                dataset_files.append(fname)

    if not dataset_files:
        print("No dataset files found in the specified directory.")
        return None, protocol_map, service_map, flag_map, label_map

    dataset_files.sort()

    # --- menu de seleção (corrigido) ---
    print("dataset encontrado")
    for idx, name in enumerate(dataset_files):
        print(f"[{idx}] {name}")

    file_idx = int(input("Qual destes queres converter para multiclass? (número): "))
    if file_idx < 0 or file_idx >= len(dataset_files):
        print("Índice inválido.")
        return None, protocol_map, service_map, flag_map, label_map

    selected_fname = dataset_files[file_idx]

    in_path = os.path.join(rootfolder, selected_fname)
    print(f"path escolhido: {in_path}")

    csv_dir = os.path.join(rootfolder, "csvfiles")
    os.makedirs(csv_dir, exist_ok=True)

    out_path = os.path.join(csv_dir, selected_fname + ".multiclass.csv")
    print(f"path final: {out_path}")

    if os.path.isfile(out_path):
        print(f"CSV já existe: {out_path}")
        return out_path, protocol_map, service_map, flag_map, label_map

    next_protocol_id = max(protocol_map.values(), default=-1) + 1
    next_service_id  = max(service_map.values(),  default=-1) + 1
    next_flag_id     = max(flag_map.values(),     default=-1) + 1

    with open(in_path, "r", errors="ignore") as f, open(out_path, "w", newline="") as out:
        writer = csv.writer(out)

        for line in f:
            cols = line.strip().split(",")
            if len(cols) < 5:
                continue

            # proto / service / flag
            proto = cols[1]
            if proto not in protocol_map:
                protocol_map[proto] = next_protocol_id
                next_protocol_id += 1
            cols[1] = protocol_map[proto]

            service = cols[2]
            if service not in service_map:
                service_map[service] = next_service_id
                next_service_id += 1
            cols[2] = service_map[service]

            flag = cols[3]
            if flag not in flag_map:
                flag_map[flag] = next_flag_id
                next_flag_id += 1
            cols[3] = flag_map[flag]

            # ----------------------------
            # attack_key (sem criar coluna nova!)
            # ----------------------------
            if choice_dataset == 1:
                if choice_split == 1:
                    # KDD labeled: label está na última coluna -> limpa no sítio
                    cols[-1] = clean_attack(cols[-1])
                    attack_key = cols[-1]
                else:
                    attack_key = "unlabeled"
            else:
                # NSL-KDD: penúltima é attack-type, última é difficulty
                cols[-2] = clean_attack(cols[-2])
                attack_key = cols[-2]

            # family (ÚNICA coluna nova)
            family_key = attack2family.get(attack_key, "unknown")

            # label_map attack -> id (se quiseres usar mais tarde)
            _ = get_or_create_id(label_map, attack_key)

            # adiciona APENAS family
            if choice_dataset == 1 and choice_split == 2:
                # KDD unlabeled: não acrescenta nada (só features encoded)
                writer.writerow(cols)
            else:
                # KDD labeled e NSL: acrescenta family
                cols.append(family_key)
                writer.writerow(cols)


    print("Conversão multiclass concluída.")
    return out_path, protocol_map, service_map, flag_map, label_map
