default: figures
	pdflatex main
	bibtex main

submission: default
	pdfjam --outfile submission.pdf -- main.pdf 1-11
	pdfjam --outfile supplementary.pdf -- main.pdf 12-

sources.zip: default
	rm -rf sources
	mkdir sources
	cp -r Makefile *.tex *.sty *.bib *.bst *.table figures sources
	sed -i.bak '/^[ ^t]*%/d' sources/*.tex
	sed -i.bak 's///' sources/Makefile
	rm sources/*.bak
	zip sources.zip -r sources

figures:
	make -C figures

clean:
	rm main.aux main.log main.out main.bbl main.blg

refdb:
	git clone https://github.com/percyliang/refdb

refdb-update: 
	make -C refdb
	cp refdb/all.bib .

run:
	go main.pdf

paper.zip:
	zip -R paper.zip figures refdb/all.bib *.tex *.sty *.tables *.table *.sty *.bst *.bbl *.blg Makefile

.PHONY: figures clean refdb paper.zip
		
