build_push_docs:
	@mkdocs build
	@mkdocs gh-deploy

build_cython:
	@python setup.py build_ext --inplace

build_package:
	@python setup.py sdist bdist_wheel

publish_package:
	@twine upload --skip-existing dist/*

git_log:
	@git log --all --graph --decorate --oneline

git_remove_cached:
	@git rm -r --cached .idea


BUNDLE_PATH := ~/.rqalpha
rqalpha_download_bundle:
	@rqalpha download-bundle -d $(BUNDLE_PATH)


# download bundle from "http://bundle.assets.ricequant.com/bundles_v4/rqbundle_202306.tar.bz2"

rqalpha_update_bundle:
	@rqalpha update-bundle

download_sample_data:
	@aria2c -s 10 https://github.com/nymath/torchquantum/releases/download/V0.1/stocks_f64.pkl.zip

# add hash

# git fetch origin --prune: The --prune option tells Git to remove any remote tracking branches that no longer exist on the remote repository.


# git branch --all
# git branch --verbose
# git branch --remotes

# git push --set-upstream origin main:main

