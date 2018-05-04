import requests
import re
import os

for page in range(3, 15):

    res = requests.get("https://www.chronext.de/kaufen?page=%d" % page)
    links = re.findall("""\"@type\":\"ListItem\",.*?\"position\":[0-9]+,.*?\"url\": \"https://www.chronext.de/([^\"]+)""", res.content.decode("utf-8"), re.DOTALL)

    for link in links:

        item_name = link.replace('/', '-')

        res = requests.get("https://www.chronext.de/%s" % link).content.decode("utf-8")
        m = re.search("""data-path="//(.*?)" data-index-start=""", res)
        datapath = m.groups()[0]

        res = requests.get("https://%s/content.json" % datapath).content.decode("utf-8")
        m = re.search("""count":([0-9]+),.*"version":"([0-9]+)""", res)

        if not m:
            print("no count and version found in %s" % res)
            continue

        count, version = m.groups()

        print("downloading images for %s..." % link)

        for deg in range(1, int(count)+1):
            if not os.path.isdir("images/%s" % item_name):
                os.mkdir("images/%s" % item_name)
            with open("images/%s/%03d" % (item_name, deg), "wb") as f:
                url = "https://%s/%03d_lo.jpg?v=%s" % (datapath, deg, version)
                f.write(requests.get(url).content)
