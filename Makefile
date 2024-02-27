JL = julia --project

default: init test

init:
	$(JL) -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
init-docs:
	$(JL) -e 'using Pkg; Pkg.activate("docs"); Pkg.develop(path="."); Pkg.instantiate(); Pkg.precompile()'
init-examples:
	$(JL) -e 'using Pkg; Pkg.activate("examples"); Pkg.develop(path="."); Pkg.instantiate(); Pkg.precompile()'

update:
	$(JL) -e 'using Pkg; Pkg.update(); Pkg.precompile()'
update-docs:
	$(JL) -e 'using Pkg; Pkg.activate("docs"); Pkg.update(); Pkg.precompile()'
update-examples:
	$(JL) -e 'using Pkg; Pkg.activate("examples"); Pkg.update(); Pkg.precompile()'

test:
	$(JL) -e 'using Pkg; Pkg.test()'

example-%:
	$(JL) -e 'using Pkg; Pkg.activate("examples"); include("examples/$*.jl")'

coverage:
	$(JL) -e 'using Pkg; Pkg.test(; coverage=true)'

servedocs:
	$(JL) -e 'using Pkg; Pkg.activate("docs"); using LiveServer; servedocs()'

clean:
	rm -rf docs/build

.PHONY: init test
