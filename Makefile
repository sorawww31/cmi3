CPU_FLAG := # 変数の宣言
ifneq ($(CPU),) #ifneqは変数が空でない場合に真
    CPU_FLAG := -f compose.CPU.yaml
endif


default: build

build:
	docker compose $(CPU_FLAG) build

bash:
	docker compose $(CPU_FLAG) run --rm $(EXP_FLAG) kaggle bash

jupyter:
	docker compose $(CPU_FLAG) up

down:
	docker compose $(CPU_FLAG) down