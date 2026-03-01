.PHONY: post

post:
	@scripts/new-post.sh "$(slug)" "$(title)" "$(tags)"
