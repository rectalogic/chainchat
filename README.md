# ChainChat - Chat with LangChain

CLI to chat with any [LangChain](https://python.langchain.com/docs/introduction/) model,
also supports [tool calling](https://python.langchain.com/docs/integrations/tools/)
and [multimodality](https://python.langchain.com/docs/concepts/multimodality/).

## Chat Models

ChainChat will introspect any installed `langchain_*` packages and make any `BaseChatModel` subclasses
available as commands with the models attributes as options - `chainchat <model-command> --<option> <value>`.

## API Keys

API keys are accessed via environment variables.
By default they are loaded from a `.env` file located in the current directory.
You can specify a different file using the `chainchat --dotenv` option.

## OpenAI Compatible Models

You can use any OpenAI compatible model with ChainChat.
For example to use [xAI Grok](https://x.ai/api) put your `XAI_API_KEY` in your `.env` file
and alias it to `OPENAI_API_KEY`:
```sh-session
$ chainchat --alias-env OPENAI_API_KEY XAI_API_KEY open-ai --model-name grok-beta --openai-api-base https://api.x.ai/v1 prompt --tool read_file  "Read and summarize the file ./LICENSE.txt"
I am reading the file ./LICENSE.txt to summarize its contents.
...
```