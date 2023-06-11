build_push_docs:
	@mkdocs build
	@mkdocs gh-deploy

build_cython:
	@python setup.py build_ext --inplace

build_package:
	@python setup.py sdist bdist_wheel
	@twine upload --skip-existing dist/*



# add hash

# git fetch origin --prune: The --prune option tells Git to remove any remote tracking branches that no longer exist on the remote repository.


# git branch --all
# git branch --verbose
# git branch --remotes