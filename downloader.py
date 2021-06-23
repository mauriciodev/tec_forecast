import requests
import sys, os
from requests.auth import HTTPBasicAuth
print("Remember that this script requires .netrc on your home folder")
requests.packages.urllib3.disable_warnings()
os.environ['OPENSSL_CONF']=os.path.join(os.getcwd(),"openssl.conf")
import logging

class magnDownloader():
    def __init__(self):
        self.url = "http://wilkilen.fcaglp.unlp.edu.ar/ion/magn/"


    def download(self,jday, year, rootdir):
        r = requests.get(self.url)
        for line in r.text.splitlines():
            f=line.split(' ')[0]
            print(line)


#Copied from https://cddis.nasa.gov/Data_and_Derived_Products/CDDIS_Archive_Access.html
class cddisDownloader():
    def __init__(self):
        self.url = "https://cddis.nasa.gov/archive/gnss/products/ionex/2021/131/"

    def listfolder(self,url):
        #Adds '*?list' to the end of URL if not included already
        if not url.endswith("*?list"):
            url = url + "*?list"

        #Makes request of URL, stores response in variable r
        r = requests.get(url, verify=False)
        res=[]
        for line in r.text.splitlines():
            f=line.split(' ')[0]
            if f[0]!="#":
                res.append(f)

        return res

    #Prints the results of the directory listing
    #print(r.text)


    def _download(self,url,rootdir):
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        
        # Assigns the local file name to the last part of the URL
        filename = url.split('/')[-1]

        # Makes request of URL, stores response in variable r
        r = requests.get(url)
        if r.status_code==404:
            logging.warning("File not found: "+url) 

        # Opens a local file of same name as remote file for writing to
        with open(os.path.join(rootdir,filename), 'wb') as fd:
            for chunk in r.iter_content(chunk_size=1000):
                fd.write(chunk)

        # Closes local file
        fd.close()
    
    def download(self,jday, year, rootdir):
        fileurl = "https://cddis.nasa.gov/archive/gnss/products/ionex/{year}/{jday:03d}/codg{jday:03d}0.{lastDigits:02d}i.Z".format(year=year,jday=jday,lastDigits=year % 100)
        filename = fileurl.split('/')[-1]
        output=os.path.join(rootdir,filename)
        if not os.path.exists(output) and not os.path.exists(output[:-2]): ##ignoring downloaded data even if it's uncompressed
            print(year, jday, fileurl)
            self._download(fileurl,rootdir)
    
    
def main():
    rootdir="ionex"
    years=[2019,2020]#[2021,2020,2019,2018]
    cddis=cddisDownloader()
    magn=magnDownloader()
    for year in years:
        leap= 0 if (2000+year)%4 else 1
        for jday in range(1,366+leap):
            cddis.download(jday,year,rootdir)
            #magn.download(jday,year,rootdir)
    os.system("uncompress ./ -r")


if __name__=="__main__":
    main()
