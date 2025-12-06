import yaml
import webbrowser
from kiteconnect import KiteConnect
from pathlib import Path

# -------------------------------------------------
# AUTO-LOCATE config.yaml (search upward)
# -------------------------------------------------
def find_config():
    cur = Path(__file__).resolve()
    root = cur.anchor  # C:\ or / on Linux/Mac

    while True:
        candidate = cur.parent / "config/config.yaml"
        if candidate.exists():
            return candidate
        if str(cur.parent) == root:
            break
        cur = cur.parent

    raise FileNotFoundError("config.yaml not found in any parent directory.")


CONFIG_PATH = find_config()


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"\n✔ Saved updated access_token to {CONFIG_PATH}")


# -------------------------------------------------
# AUTO-LOCATE .env (search upward)
# -------------------------------------------------
def find_env():
    cur = Path(__file__).resolve()
    root = cur.anchor

    while True:
        candidate = cur.parent / ".env"
        if candidate.exists():
            return candidate
        if str(cur.parent) == root:
            break
        cur = cur.parent

    raise FileNotFoundError(".env file not found in any parent directory.")


# -------------------------------------------------
# Update ACCESS_TOKEN in .env
# -------------------------------------------------
def update_env_token(new_token):
    try:
        env_path = find_env()
    except FileNotFoundError:
        print("⚠ .env file not found. Skipping .env update.")
        return

    lines = []
    token_written = False

    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith("ACCESS_TOKEN="):
                    lines.append(f"ACCESS_TOKEN={new_token}\n")
                    token_written = True
                else:
                    lines.append(line)

    if not token_written:
        lines.append(f"ACCESS_TOKEN={new_token}\n")

    with open(env_path, "w") as f:
        f.writelines(lines)

    print(f"✔ Updated access_token in {env_path}")


# -------------------------------------------------
# MAIN REFRESH LOGIC
# -------------------------------------------------
def refresh_token():
    cfg = load_config()


    api_key = cfg["api_key"]
    api_secret = cfg["api_secret"]

    print("\n==============================")
    print("   Zerodha Token Refresher")
    print("==============================")

    print(f"\nUsing API Key: {api_key}")
    print("Opening Zerodha login page...\n")

    login_url = f"https://kite.trade/connect/login?api_key={api_key}"
    webbrowser.open(login_url)

    print("After logging in, copy ONLY the request_token from the URL:")
    print("Example:")
    print("  https://your-redirect-url/?request_token=XXXXX&action=login\n")

    request_token = input("Paste request_token here: ").strip()

    print("\nExchanging request_token for access_token...")

    kite = KiteConnect(api_key=api_key)
    data = kite.generate_session(request_token, api_secret=api_secret)

    access_token = data["access_token"]
    print(f"\n✔ New access_token: {access_token}")

    # Save to config.yaml
    cfg["access_token"] = access_token
    save_config(cfg)

    # Save to .env
    update_env_token(access_token)

    print("\n==============================")
    print("  ✔ TOKEN REFRESH COMPLETE")
    print("==============================\n")

    return access_token


if __name__ == "__main__":
    refresh_token()
