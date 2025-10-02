# pip install youtube_dl
# python -m pip install -U yt-dlp certifi

# Disclaimer: It is illegal to download copyrighted content from youtube. 
# However, for content that is within the law, you can use python to download 
# many youtube videos at once in case you want to watch them later offline:

# Asks for the url

from yt_dlp import YoutubeDL

link = input("Write the url: ")
with YoutubeDL({}) as ydl:
    ydl.download([link])


# # Instantiate the YoutubeDL class from the youtube_dl framework
# ydl = youtube_dl.YoutubeDL({})
# # Download the video.
# ydl.download([link])