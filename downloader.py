import requests
import sys, os, subprocess
from requests.auth import HTTPBasicAuth
print("Remember that this script requires .netrc on your home folder")
requests.packages.urllib3.disable_warnings()
os.environ['OPENSSL_CONF']=os.path.join(os.getcwd(),"openssl.conf")
import logging
import urllib.request
import netrc

def download(url,destination):
    os.makedirs(destination,exist_ok=True)
    outfile=url.split("/")[-1]
    outfile=os.path.join(destination,outfile)
    if not os.path.exists(outfile) or update==True:
        print(url, " -> ", outfile)
        urllib.request.urlretrieve(url,outfile)
    return outfile



#Copied from https://cddis.nasa.gov/Data_and_Derived_Products/CDDIS_Archive_Access.html
class cddisDownloader():
    def __init__(self):
        pass
    
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
    
    def _download2(self,url,rootdir):
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        filename = url.split('/')[-1]
        os.chdir(rootdir)
        cmd=f"curl -c [file] -n -L -O \"{url}\""
        if not os.path.exists(filename):
            subprocess.run(cmd, shell=True)
    
    def download(self,jday, year, rootdir, prefix="codg"):
        fileurl = "https://cddis.nasa.gov/archive/gnss/products/ionex/{year}/{jday:03d}/{prefix}{jday:03d}0.{lastDigits:02d}i.Z".format(year=year,jday=jday,lastDigits=year % 100, prefix=prefix)
        filename = fileurl.split('/')[-1]
        output=os.path.join(rootdir,filename)
        if not os.path.exists(output) and not os.path.exists(output[:-2]): ##ignoring downloaded data even if it's uncompressed
            print(year, jday, fileurl)
            try:
                self._download2(fileurl,rootdir)
            except:
                print(f"Failed to download {fileurl}.")
    
class magnDownloader():
    def __init__(self):
        top_level_url = "wilkilen.fcaglp.unlp.edu.ar"
        netrcData = netrc.netrc()
        authTokens = netrcData.authenticators(top_level_url)
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        top_level_url="http://"+top_level_url
        password_mgr.add_password(None, top_level_url, authTokens[0], authTokens[2])
        handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
        # create "opener" (OpenerDirector instance)
        self.opener = urllib.request.build_opener(handler)
        # use the opener to fetch a URL
        a_url="http://wilkilen.fcaglp.unlp.edu.ar/"
        self.opener.open(a_url)
        # Install the opener.
        # Now all calls to urllib.request.urlopen use our opener.
        urllib.request.install_opener(self.opener)
        
    def download(self,jday, year, rootdir, prefix="magn"):
        fileurl = "http://wilkilen.fcaglp.unlp.edu.ar/ion/magn/{year}/{jday:03d}/{prefix}{jday:03d}0.{lastDigits:02d}i.Z".format(year=year,jday=jday,lastDigits=year % 100, prefix=prefix)
        filename = fileurl.split('/')[-1]
        output=os.path.join(rootdir,filename)
        if not os.path.exists(output) and not os.path.exists(output[:-2]): ##ignoring downloaded data even if it's uncompressed
            print(year, jday, fileurl)
            try:
                urllib.request.urlretrieve(fileurl, output)
            except:
                print(f"Failed to download {fileurl}.")
def main():
    rootdir="ionex"
    years=range(2012,2022)#[2021,2020,2019,2018]
    cddis=cddisDownloader()
    magn=magnDownloader()
    
    for year in years:
        leap= 0 if (2000+year)%4 else 1
        for jday in range(1,366+leap):
            cddis.download(jday,year,rootdir)
    years=[2019,2020,2015]#[2021,2020,2019,2018]    
    for year in years:
        leap= 0 if (2000+year)%4 else 1
        for jday in range(1,366+leap):
            cddis.download(jday,year,rootdir,prefix="c1pg")
            #cddis.download(jday,year,rootdir,prefix="corg")
            #magn.download(jday,year,rootdir)
    os.chdir(rootdir)
    os.system("uncompress *.Z")


if __name__=="__main__":
    main()
