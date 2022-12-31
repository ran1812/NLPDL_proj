#!/bin/bash

# downloads all MEDLINE/Pubmed citations in the annual baseline.
cd data/

for i in {1100..1165}; do
    fname="1"
    if ((i < 10)); then
        fname="ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed23n000$i.xml.gz"
    elif ((i < 100)); then
        fname="ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed23n00$i.xml.gz"
    elif ((i < 1000)); then
        fname="ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed23n0$i.xml.gz"
    elif ((i < 10000)); then
        fname="ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed23n$i.xml.gz"
    fi
    echo $fname;
    wget $fname;
    sleep 1;
done

for i in {1100..1165}; do
    fname="1"
    if ((i < 10)); then
        fname="pubmed23n000$i.xml.gz"
    elif ((i < 100)); then
        fname="pubmed23n00$i.xml.gz"
    elif ((i < 1000)); then
        fname="pubmed23n0$i.xml.gz"
    elif ((i < 10000)); then
        fname="pubmed23n$i.xml.gz"
    fi
    gzip -d $fname
done