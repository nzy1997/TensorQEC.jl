JL = julia --project

default: init test

init:
	$(JL) -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
init-docs:
	$(JL) -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

update:
	$(JL) -e 'using Pkg; Pkg.update(); Pkg.precompile()'
update-docs:
	$(JL) -e 'using Pkg; Pkg.activate("docs"); Pkg.update(); Pkg.precompile()'

test:
	$(JL) -e 'using Pkg; Pkg.test()'

coverage:
	$(JL) -e 'using Pkg; Pkg.test(; coverage=true)'

servedocs:
	$(JL) -e 'using Pkg; Pkg.activate("docs"); using LiveServer; servedocs()'

clean:
	rm -rf docs/build

.PHONY: init test
