"""Shared preprocessing for training (train_model.py) and the Streamlit app (app.py)."""
import numpy as np
import pandas as pd

CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
CAT = ["protocol_type", "service", "flag"]
FEATURES = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
            "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
            "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
            "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
            "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
            "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
RAW_COLUMNS = FEATURES + ["label", "difficulty"]
CONSTANT = ["num_outbound_cmds"]
R2L_BOOST = 5   # R2L probability is multiplied by this before picking the class (see README, "R2L boost")

# Login / privilege features that R2L and U2R depend on: never removed by the correlation filter
PROTECTED = ["hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "is_host_login", "is_guest_login"]

ENGINEERED_OLD = ["total_bytes", "bytes_ratio", "log_src_bytes", "log_dst_bytes", "log_duration",
                  "failed_login_flag", "srv_count_ratio", "error_rate_sum", "dst_host_error_sum"]
ENGINEERED_NEW = ["privilege_activity", "file_activity", "login_risk", "failed_no_login",
                  "content_activity", "compromised_flag", "bytes_per_second"]

_ATTACKS = {
    "DoS": "back land neptune pod smurf teardrop apache2 mailbomb processtable udpstorm worm",
    "Probe": "ipsweep nmap portsweep satan mscan saint",
    "R2L": "ftp_write guess_passwd imap multihop phf spy warezclient warezmaster sendmail named "
           "snmpgetattack snmpguess xlock xsnoop httptunnel",
    "U2R": "buffer_overflow loadmodule perl rootkit ps sqlattack xterm"}
ATTACK_MAP = {"normal": "Normal"}
for _cls, _names in _ATTACKS.items():
    for _a in _names.split():
        ATTACK_MAP[_a] = _cls


def clean(df):
    """Trim/lowercase text, drop duplicates, drop 'difficulty', map attack names to 5 classes."""
    df = df.drop_duplicates().reset_index(drop=True)
    for c in CAT + ["label"]:
        df[c] = df[c].astype(str).str.strip().str.lower()
    df = df.drop(columns="difficulty")
    df["attack_class"] = df["label"].map(ATTACK_MAP)
    return df


def engineer(df, new=True):
    d = df.copy()
    d["total_bytes"] = d.src_bytes + d.dst_bytes
    d["bytes_ratio"] = d.src_bytes / (d.dst_bytes + 1)
    d["log_src_bytes"] = np.log1p(d.src_bytes)
    d["log_dst_bytes"] = np.log1p(d.dst_bytes)
    d["log_duration"] = np.log1p(d.duration)
    d["failed_login_flag"] = (d.num_failed_logins > 0).astype(int)
    d["srv_count_ratio"] = d.srv_count / (d["count"] + 1)
    d["error_rate_sum"] = d.serror_rate + d.rerror_rate
    d["dst_host_error_sum"] = d.dst_host_serror_rate + d.dst_host_rerror_rate
    if new:
        d["privilege_activity"] = d.root_shell + d.su_attempted + d.num_root + d.num_shells
        d["file_activity"] = d.num_file_creations + d.num_access_files
        d["login_risk"] = d.num_failed_logins + d.is_guest_login + d.is_host_login
        d["failed_no_login"] = ((d.num_failed_logins > 0) & (d.logged_in == 0)).astype(int)
        d["content_activity"] = d.hot + d.num_compromised + d.privilege_activity + d.file_activity
        d["compromised_flag"] = (d.num_compromised > 0).astype(int)
        d["bytes_per_second"] = d.total_bytes / (d.duration + 1)
    return d


def build_matrix(df, dummy_cols, drop=()):
    """Numeric columns + one-hot columns (same columns for train, test and app input)."""
    skip = set(CAT) | {"label", "attack_class"} | set(drop)
    num = [c for c in df.columns if c not in skip]
    dummies = pd.get_dummies(df[CAT]).reindex(columns=dummy_cols, fill_value=0)
    return pd.concat([df[num].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1).astype(float)
