#!/usr/bin/env python
# coding: utf-8


import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
from urllib import parse
import csv
import argparse
from time import sleep


Errs = []


Snum = 37257
Sdate = date(2002, 1, 1)


date.fromisoformat("2002-02-01")


def getDate(num): return Sdate + timedelta(days=num-Snum)


date.fromisoformat("2002-02-01") - date.fromisoformat("2002-05-01")


def getNum(datestr): return (date.fromisoformat(datestr) - Sdate).days + Snum


getDate(39218)


getNum("2007-05-16")


def getlinks(startdate):
    baseurl = "https://navbharattimes.indiatimes.com"
    relurl = f"/archivelist/starttime-{startdate}.cms"
    url = baseurl+relurl
    resp = requests.get(url)
    htmltext = ""
    if(resp.status_code == 200):
        htmltext = resp.text
        soup = BeautifulSoup(htmltext, "html.parser")
        tables = soup.select("table")
        if(len(tables) > 0):
            tlinks = tables[0].find_all('a')
            links = [baseurl+t["href"] for t in tlinks]
            return links
        else:
            if(not htmltext):
                Errs.append(["getlink", "empty response", url])
            else:
                Errs.append(["getlink", "no tables found", url])
    else:
        Errs.append(["getlink", f"resp-code:{resp.status_code}", url])


def getheadandbody(url):
    artresp = requests.get(url)
    if(artresp.status_code != 200):
        Errs.append(["getheadandbody", f"resp-code{artresp.status_code}", url])
    elif(not artresp.text):
        Errs.append(["getheadandbody", "empty response", url])
    elif(artresp.url == "https://navbharattimes.indiatimes.com/astro.cms"):
        pass
    else:
        try:
            soup = BeautifulSoup(artresp.text, "html.parser")
            head = soup.select(".story-article h1")[0].get_text()
            body = soup.select(".story-content")[0].get_text()
            author = "NA | NA"
            alist = [x.get_text()
                     for x in soup.findAll("span", {"itemprop": "author"})]
            if len(alist) >= 2:
                author = f"{alist[0]} | {alist[1]}"
            plist = parse.urlparse(artresp.url).path
            index = plist.find("/articleshow")
            if(index != -1):
                return [plist[:index], author, head, body]
            return ["NA", author, head, body]
        except:
            with open("pg.html", "w", encoding='utf-8') as f:
                f.write(artresp.text)
            Errs.append(["getheadandbody", "body not found", url])


def fetchDateHelper(numdate):
    links = getlinks(numdate)
    newsdate = getDate(numdate)
    filename = f"{newsdate.year}-{newsdate.month}"
    print(f"Fetching news from date {newsdate}, file={filename}.csv", end="\r")
    with open(f"data/{filename}.csv", "a", newline='', encoding='utf-8') as fdata:
        writer = csv.writer(fdata)
        for link in links:
            hb = getheadandbody(link)
            if hb and len(hb) >= 4:
                writer.writerow([newsdate, link, *hb])


sleeptime = 1


def fetchDate(numdate):
    global sleeptime
    try:
        fetchDateHelper(numdate)
        sleeptime = 1
    except Exception as e:
        print(e)
        print(f"Sleeping for {sleeptime} seconds...")
        sleep(sleeptime)
        sleeptime *= 2
        if(sleeptime < 2**15):
            fetchDate(numdate)
        else:
            pass


if __name__ == "__main__":
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument("startdate",default="2002-01-01",
    #         help="Enter the date from which you want to start fetching news. Format is 'yyyy-mm-dd'")
    #     args = parser.parse_args()
    #     snum = getNum(args.startdate)
    print("Enter the date from which you want to start fetching news. Format is 'yyyy-mm-dd'")
    print("Example:  2002-01-01")
    dstr = input()
    snum = getNum(dstr)
    for i in range(snum, 44583):
        fetchDate(i)


print(Errs)
with open("Errs.txt", "w", encoding='utf-8') as ferr:
    print(*Errs, sep="\n", file=ferr)
