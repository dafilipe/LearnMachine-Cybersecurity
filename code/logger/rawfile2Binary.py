import os
import csv

def rawfile2Binary(rootfolder, choice, choicelabel, protocol_map=None, service_map=None, flag_map=None):
    files = os.listdir(rootfolder)

    if protocol_map is None:
        protocol_map = {}
    if service_map is None:
        service_map = {}
    if flag_map is None:
        flag_map = {}

    next_protocol_id = max(protocol_map.values(), default=-1) + 1
    next_service_id  = max(service_map.values(),  default=-1) + 1
    next_flag_id     = max(flag_map.values(),     default=-1) + 1

    kdd_files = []

    # ---- KDDCup99 ----
    if choice == 1:
        for fname in files:
            fullpath = os.path.join(rootfolder, fname)
            if not (fname.startswith("kddcup") and os.path.isfile(fullpath)):
                continue

            if choicelabel == 1:
                # labeled: corrected e não unlabeled
                if ("unlabeled" not in fname) and ("corrected" in fname):
                    kdd_files.append(fname)

            elif choicelabel == 2:
                # unlabeled
                if "unlabeled" in fname:
                    kdd_files.append(fname)

    # ---- NSL-KDD ----
    else:
        for fname in files:
            fullpath = os.path.join(rootfolder, fname)
            if not os.path.isfile(fullpath):
                continue

            if choicelabel == 1 and fname.startswith("KDDTrain"):
                kdd_files.append(fname)

            elif choicelabel == 2 and fname.startswith("KDDTest"):
                kdd_files.append(fname)

    if len(kdd_files) == 0:
        print("No kddcup files found in the specified directory.")
        return None, protocol_map, service_map, flag_map

    kdd_files.sort()

    print("dataset encontrado")
    for idx, name in enumerate(kdd_files):
        print(f"[{idx}] {name}")

    file_idx = int(input("Qual destes queres tornar em binary? (número): "))
    if file_idx < 0 or file_idx >= len(kdd_files):
        print("Índice inválido.")
        return None, protocol_map, service_map, flag_map

    file = kdd_files[file_idx]
    finalpath = os.path.join(rootfolder, file)
    print(f"path escolhido: {finalpath}")

    csv_dir = os.path.join(rootfolder, "csvfiles")
    os.makedirs(csv_dir, exist_ok=True)

    out_path = os.path.join(csv_dir, file + ".csv")
    print(f"path final: {out_path}")

    if os.path.isfile(out_path):
        print(f"CSV já existe: {out_path}")
        return out_path, protocol_map, service_map, flag_map

    with open(finalpath, "r", errors="ignore") as f, open(out_path, "w", newline="") as out:
        print("Convertendo para binary e salvando em CSV...")
        writer = csv.writer(out)

        for line in f:
            columns = line.strip().split(",")
            if len(columns) < 4:
                continue

            # proto/service/flag mapping 
            proto = columns[1]
            if proto not in protocol_map:
                protocol_map[proto] = next_protocol_id
                next_protocol_id += 1
            columns[1] = protocol_map[proto]

            service = columns[2]
            if service not in service_map:
                service_map[service] = next_service_id
                next_service_id += 1
            columns[2] = service_map[service]

            flag = columns[3]
            if flag not in flag_map:
                flag_map[flag] = next_flag_id
                next_flag_id += 1
            columns[3] = flag_map[flag]

            # labels
            if choice == 1:
                # KDDCup99: só labeled tem label no fim
                if choicelabel == 1:
                    label = columns[-1].strip()
                    columns[-1] = 0 if label in ("normal", "normal.") else 1

            elif choice == 2:
                label = columns[-2].strip()
                columns[-2] = 0 if label in ("normal", "normal.") else 1
                # difficulty (columns[-1]) fica intacta

            writer.writerow(columns)

    print("Successfully returned file :)")
    return out_path, protocol_map, service_map, flag_map
