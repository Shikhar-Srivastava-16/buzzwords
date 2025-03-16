from atproto import Client
import re, json, requests

client = Client()

nope = client.login(
    login='footloose-mini.bsky.social', 
    password='1234'
)

did_dict = {
    "AOC": 'did:plc:p7gxyfr5vii5ntpwo7f6dhe2',
    "Fetterman": 'did:plc:hmy4bopsymtjikfq5i5mekn6',
    "gtakei": 'did:plc:y4zs4cabaezzwx3bz2e5nnj2',
    "chrishayes": 'did:plc:e62gb2ushvtvjvqcbrxeaw2n',
    "aaronrm": 'did:plc:2vtbmhmrwzbqcfv4we4uxzzt',
    "jbouie": 'did:plc:nvfposmpmhegtyvhbs75s3pw',
    "chuck": 'did:plc:zlgotdjaxzuhw4nefvkxxvxa',
    "repcasar": 'did:plc:3kapv2cmc5r6ehswf6nwo5wt'
}

# while True:
#     try:
#         handle = str(input("What is the user's handle? "))
#         name = str(input("What is the user's Name? "))
#         url_name = f"https://bsky.social/xrpc/com.atproto.identity.resolveHandle?handle={handle}"
#         resp = requests.get(url=url_name)
#         did_dict[name] = (resp.json().get("did"))
    
#     except:
#         break

def get_first_100_posts(did: str, list_of_tweets: list, tinyFile):
    print("-----------------------{did}-----------------------------")
    # data = client.get_author_feed(
    #     actor=did,
    #     filter='posts_and_author_threads',
    #     limit=100,
    # )

    cursor = ''
    all_posts = []

    # Fetch posts in batches
    while len(list_of_tweets) < 1000:
        print("loop")
        # Make the API call to fetch posts from the author
        # should run up to 10 times for 8 people
        data = client.get_author_feed(
            actor=did,
            filter='posts_and_author_threads',  # Or use 'posts_no_replies', depending on your needs
            cursor=cursor,
            limit=100
        )

        # Add the new posts to the list
        # all_posts.extend(data.feed)
        
        # If we have more posts, get the cursor for the next batch
        cursor = data.cursor
        
        # If no cursor is returned, it means we've reached the end of the feed
        if not cursor:
            break

        for f in data.feed:

            # Assuming f.post.record.text is the input tweet
            cleaned_tweet = f.post.record.text
            tinyFile.write(f"{cleaned_tweet}\n")

            # Remove special characters
            cleaned_tweet = re.sub(r'[^A-Za-z0-9\s\'\.]', '', cleaned_tweet)

            # Remove escape sequences like \n, \t, etc.
            cleaned_tweet = re.sub(r'[\n\t\r]', ' ', cleaned_tweet)

            # Remove all words with 1, 2, or 3 letters
            cleaned_tweet = re.sub(r'\b\w{1,3}\b', '', cleaned_tweet)

            # Optionally, clean up extra spaces after removing the short words
            cleaned_tweet = re.sub(r'\s+', ' ', cleaned_tweet).strip()

            list_of_tweets += [
                (
                    cleaned_tweet,
                    f.post.record.created_at[0:10]
                )
            ]

def add_to_json_list(tweet_data, json_list):
    # Ensure tweet_data is a tuple with the format (tweet_text, date)
    if isinstance(tweet_data, tuple) and len(tweet_data) == 2:
        tweet, date = tweet_data
        # Create a dictionary for the tweet
        tweet_dict = {
            "tweet": tweet,
            "date": date
        }
        # Append the tweet to the json list
        json_list.append(tweet_dict)
        return json_list
    else:
        raise ValueError("tweet_data must be a tuple with (tweet_text, date)")
    
tweets = []
json_list = []
str_output = ""
# https://bsky.social/xrpc/com.atproto.identity.resolveHandle?handle=aoc.bsky.social


with open("../data/tinybsky.txt", "w") as file:
    for key in did_dict:
        get_first_100_posts(did_dict[key], tweets, file)

for tweet_data in tweets:
    json_list = add_to_json_list(tweet_data, json_list)

json_output = json.dumps(json_list, indent=4)

    # Print the output (or save to a file)
print(json_output)

for tweet_data in tweets:
    json_list = add_to_json_list(tweet_data, json_list)

    # Convert the list to a JSON string (optional, if you want to save it as a JSON file)
    json_output = json.dumps(json_list, indent=4)

with open("../data/bsky.json", "w") as file:
    file.write(json_output)

