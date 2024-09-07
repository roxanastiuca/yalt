import argparse
import requests
import re


def get_popular_repositories(language=None, per_page=100, page=10):
    base_url = "https://api.github.com/search/repositories"
    query = "stars:>1"  # Search for repositories with more than 1 star
    if language:
        query += f" language:{language}"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": per_page,
        "page": page
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.json()

def extract_repo_name(github_url):
    pattern = r"https://github\.com/([^/]+/[^/]+)"
    match = re.match(pattern, github_url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid GitHub URL")

def get_latest_release_files(repo_name, release_name):
    api_url = f"https://api.github.com/repos/{repo_name}/releases/latest"
    if release_name != "latest":
        api_url = f"https://api.github.com/repos/{repo_name}/releases/tags/{release_name}"
    response = requests.get(api_url)
    response.raise_for_status()
    release_data = response.json()
    asset_files = [asset['name'] for asset in release_data['assets']]
    return asset_files

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--github_url", help="GitHub repository URL")
    group.add_argument("--all", help="Get all packages", action="store_true")
    parser.add_argument("--release", help="Release version")
    args = parser.parse_args()

    if args.all:
        popular_repositories = []
        try:
            popular_repositories = get_popular_repositories()
        except Exception as e:
            print(f"Error: {e}")
            exit(1)
        
        
        fout = open("popular_repositories.txt", "a")
        for repo in popular_repositories['items']:
            try:
                files = get_latest_release_files(repo['full_name'], "latest")
            except Exception as e:
                print(f"Error: {e}")
                continue
            print(f"{repo['full_name']} - {repo['description']}")
            if files:
                files_str = ",".join(files)
                fout.write(f"{repo['full_name']} {files_str} 0 \"{repo['description']}\"\n")
                print("Files in the latest release:")
                for file in files:
                    print(f"- {file}")
            else:
                print("No files found in the latest release.")

        fout.close()
        exit(0)

    try:
        repo_name = extract_repo_name(args.github_url)
        print(f"Repository Name: {repo_name}")
        files = get_latest_release_files(repo_name, args.release if args.release else "latest")
        if files:
            print("Files in the latest release:")
            for file in files:
                print(f"- {file}")
        else:
            print("No files found in the latest release.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
