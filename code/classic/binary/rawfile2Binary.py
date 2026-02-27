import os
import csv


def clean_attack(x: str) -> str:
    x = (x or "").strip()
    if x.endswith("."):
        x = x[:-1]
    return x


def attack_to_binary_label(attack_key: str) -> int:
    # normal -> 0 ; qualquer outro -> 1
    return 0 if attack_key == "normal" else 1


def rawfile2Binary(
    rootfolder,
    choice: int,        # 1=KDDCup99, 2=NSL-KDD
    choicelabel: int,   # KDD: 1=labeled 2=unlabeled ; NSL: 1=train 2=test
    protocol_map=None,
    service_map=None,
    flag_map=None,
):
    files = os.listdir(rootfolder)

    if protocol_map is None:
        protocol_map = {}
    if service_map is None:
        service_map = {}
    if flag_map is None:
        flag_map = {}

    dataset_files = []

    # ----------------------------
    # KDDCup99
    # ----------------------------
    if choice == 1:
        for fname in files:
            fullpath = os.path.join(rootfolder, fname)
            if not (fname.startswith("kddcup") and os.path.isfile(fullpath)):
                continue

            if choicelabel == 1:
                # labeled: normalmente "corrected"
                if ("unlabeled" not in fname) and ("corrected" in fname):
                    dataset_files.append(fname)
            elif choicelabel == 2:
                if "unlabeled" in fname:
                    dataset_files.append(fname)

    # ----------------------------
    # NSL-KDD
    # ----------------------------
    else:
        for fname in files:
            fullpath = os.path.join(rootfolder, fname)
            if not os.path.isfile(fullpath):
                continue

            if choicelabel == 1 and fname.startswith("KDDTrain"):
                dataset_files.append(fname)
            elif choicelabel == 2 and fname.startswith("KDDTest"):
                dataset_files.append(fname)

    if not dataset_files:
        print("No dataset files found in the specified directory.")
        return None, protocol_map, service_map, flag_map

    dataset_files.sort()

    print("dataset encontrado")
    for idx, name in enumerate(dataset_files):
        print(f"[{idx}] {name}")

    file_idx = int(input("Qual destes queres converter para binary? (número): "))
    if file_idx < 0 or file_idx >= len(dataset_files):
        print("Índice inválido.")
        return None, protocol_map, service_map, flag_map

    selected_fname = dataset_files[file_idx]
    in_path = os.path.join(rootfolder, selected_fname)
    print(f"path escolhido: {in_path}")

    csv_dir = os.path.join(rootfolder, "csvfiles")
    os.makedirs(csv_dir, exist_ok=True)

    out_suffix = ".binary.csv" if (choice != 1 or choicelabel == 1) else ".binary.unlabeled.csv"
    out_path = os.path.join(csv_dir, selected_fname + out_suffix)
    print(f"path final: {out_path}")

    if os.path.isfile(out_path):
        print(f"CSV já existe: {out_path}")
        return out_path, protocol_map, service_map, flag_map

    next_protocol_id = max(protocol_map.values(), default=-1) + 1
    next_service_id  = max(service_map.values(),  default=-1) + 1
    next_flag_id     = max(flag_map.values(),     default=-1) + 1

    with open(in_path, "r", errors="ignore") as f, open(out_path, "w", newline="") as out:
        print("Convertendo para binary e salvando em CSV...")
        writer = csv.writer(out)

        for line in f:
            cols = line.strip().split(",")
            if len(cols) < 5:
                continue

            # proto/service/flag (colunas 1,2,3 no KDD/NSL raw)
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
            # KDDCup99
            # ----------------------------
            if choice == 1:
                if choicelabel == 1:
                    # labeled: última coluna é label original (ex: "neptune.")
                    attack_key = clean_attack(cols[-1])
                    y = attack_to_binary_label(attack_key)

                    # remover a label do fim e anexar y (0/1)
                    feats = cols[:-1]
                    feats.append(y)
                    writer.writerow(feats)

                else:
                    # unlabeled: NÃO há label confiável -> escrever só features
                    # (mantém tudo como veio: o ficheiro unlabeled já não tem a label no fim)
                    writer.writerow(cols)

            # ----------------------------
            # NSL-KDD
            # ----------------------------
            else:
                # NSL: penúltima é attack-type, última é difficulty
                attack_key = clean_attack(cols[-2])
                diff = cols[-1]
                y = attack_to_binary_label(attack_key)

                # features = tudo menos (attack, diff)
                feats = cols[:-2]
                feats.append(y)      # label binary
                feats.append(int(diff))  # difficulty
                writer.writerow(feats)

    print("Conversão binary concluída.")
    return out_path, protocol_map, service_map, flag_map
