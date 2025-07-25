import os
import signal
import time
import warnings
import toml
import vertexai
import vertexai.preview.generative_models as generative_models
from google.api_core import exceptions as google_exceptions
from openai import OpenAI
# from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models._generative_models import Content, Part
from anthropic import AnthropicVertex
import pprint
from openai import RateLimitError, APIConnectionError, InternalServerError
import transformers
import torch

pipeline = None

models = ['gpt-4o', 'gpt-4', 'gpt-4-turbo', 'gemini-1.5-pro-001', 'gemini-1.5-flash-001', 'gemini-1.5-flash-002', 'gemini-1.0-pro-002', 'gpt-4o-2024-08-06',
          'claude-3-opus@20240229', 'claude-3-sonnet@20240229', 'claude-3-haiku@20240307', 'Qwen/Qwen2.5-3B-Instruct', 'mlx-community/Qwen2.5-3B-Instruct-bf16',
          'Qwen/Qwen3-4B', 'Qwen/Qwen2.5-3B-Instruct', 'gemini-2.5-flash-preview-04-17', 'gemini-2.0-flash-001', 'Qwen/Qwen3-30B-A3B']

config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')

prev_model = None
api_key_params, parameters = None, None

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution exceeded time limit")


def run_with_timeout(func, chat_history, prompt, timeout):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(chat_history, prompt)
    except TimeoutException as e:
        raise e
    finally:
        signal.alarm(0)  # Cancel the alarm
    return result


def execute_prompt(*, chat_history, prompt, system_instruction, model, vertex_ai_max_attempts=10):
    global pipeline

    def read_params(model_):
        global prev_model, api_key_params, parameters
        if prev_model != model_:
            with open(os.path.join(config_dir, 'api_keys.toml'), 'r') as params:
                api_key_params = toml.load(params)
                os.environ['OPENAI_API_KEY'] = api_key_params['open_ai']['api_key']
                vertexai.init(project=api_key_params['vertex_ai']['project'],
                              location=api_key_params['vertex_ai']['location'])
            try:
                with open(os.path.join(config_dir, model + '.toml'), 'r') as params:
                    parameters = toml.load(params)
            except FileNotFoundError:
                pass
            pprint.pprint(parameters)
        prev_model = model_

    read_params(model)

    if model not in models:
        raise Exception("Invalid model. Choose from {}".format(models))
    if "gpt" in model:
        client = OpenAI()
        _chat_history = list()
        _chat_history.append({"role": "system", "content": system_instruction})
        for ch in chat_history:
            # if ch['role'] == "assistant":  # Conversion between GPT and GEMINI
            #     ch['role'] = 'system'
            if 'content' not in ch:
                ch['content'] = ch['text']
            _chat_history.append({"role": ch["role"], "content": ch["content"]})
        _chat_history.append({"role": "user", "content": prompt})
        while True:
            try:
                response = client.chat.completions.create(
                    # instructions=system_instruction,
                    max_tokens=128,
                    model=model,
                    # logprobs=True,
                    # top_logprobs=3,
                    temperature=0,
                    messages=[*_chat_history]
                )
                break
            except (RateLimitError, APIConnectionError, InternalServerError) as e:
                print(e)
                time.sleep(20)
        res_dump = response.model_dump()
        return res_dump['choices'][0]['message']['content']

    elif "Qwen" in model:
        # # Aphrodite Engine
        # openai_api_key = "EMPTY"
        # openai_api_base = "http://localhost:2242/v1"

        # vLLM
        openai_api_base = "http://localhost:8000/v1"
        openai_api_key = "token-abc123"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        _chat_history = list()
        _chat_history.append({"role": "system", "content": system_instruction})
        for ch in chat_history:
            # if ch['role'] == "assistant":  # Conversion between GPT and GEMINI
            #     ch['role'] = 'system'
            if 'content' not in ch:
                ch['content'] = ch['text']
            _chat_history.append({"role": ch["role"], "content": ch["content"]})
        _chat_history.append({"role": "user", "content": prompt})
        while True:
            try:
                response = client.chat.completions.create(
                    # instructions=system_instruction,
                    max_tokens=128,
                    model=model,
                    # logprobs=True,
                    # top_logprobs=3,
                    temperature=0,
                    messages=[*_chat_history],
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                break
            except (RateLimitError, APIConnectionError, InternalServerError) as e:
                print(e)
                time.sleep(20)
        res_dump = response.model_dump()
        return res_dump['choices'][0]['message']['content']



    elif "claude" in model:
        client = AnthropicVertex(region=api_key_params['vertex_ai']['location'],
                                 project_id=api_key_params['vertex_ai']['project'])

        _chat_history = list()
        for ch in chat_history:
            if 'content' not in ch:
                ch['content'] = ch['text']
            _chat_history.append({"role": ch["role"], "content": ch["content"]})
        _chat_history.append({"role": "user", "content": prompt})

        message = client.messages.create(
            max_tokens=512,
            # system=system_instruction,
            messages=[{"role": "user", "content": "Hello, Claude!"}],  # <-- user prompt
            model=model,
        )
        return message

    elif "gemini" in model:
        def run(chat_history_, prompt_):
            global parameters
            nonlocal safety_settings
            chat = model.start_chat(history=chat_history_, response_validation=False)
            responses_ = chat.send_message(prompt_, stream=False, safety_settings=safety_settings,
                                           generation_config=parameters['parameters'])
            return responses_

        # TODO(developer): Update and un-comment below line
        # project_id = "PROJECT_ID"

        # chat_history =
        model = GenerativeModel(model_name=model,
                                system_instruction=system_instruction)
        safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE}
        # generation_config = {"max_output_tokens": 128}
        # model._system_config = "Count the tokens:"
        # attempts = 10
        for attempt in range(vertex_ai_max_attempts):
            _chat_history = list()
            for ch in chat_history:
                if ch['role'] == "assistant":  # Conversion between GPT and GEMINI
                    ch['role'] = 'model'
                if 'content' not in ch:
                    ch['content'] = ch['text']
                _chat_history.append(Content(role=ch['role'], parts=[Part.from_text(ch['content'])]))
            try:
                assert len(_chat_history) == len(chat_history)
            except AssertionError as e:
                print(e)
            try:
                responses = run_with_timeout(run, chat_history=_chat_history, prompt=prompt, timeout=30)
                # chat = model.start_chat(history=_chat_history, response_validation=False)
                # responses = chat.send_message(prompt, stream=False, safety_settings=safety_settings,
                #                               generation_config=generation_config)
            except google_exceptions.InvalidArgument as e:
                print(e)
                break
            except TimeoutException as e:
                print(e)
                print("waiting...")
                time.sleep(243)
                continue
            except (google_exceptions.ResourceExhausted, google_exceptions.FailedPrecondition, IndexError,
                    google_exceptions.InternalServerError) as e:
                print(e)
                print("waiting...")
                time.sleep(312)
                continue
            res = responses.to_dict()['candidates'][0]
            if res['finish_reason'] == 'MAX_TOKENS':
                print(responses.text)
                # raise RuntimeError(f"Max tokens reached: {res['finish_reason']}. Set longer output.")
                return "[]"
            elif res['finish_reason'] == 'OTHER':
                time.sleep(25)
                continue
            break
        else:
            warnings.warn("Exceeded times of attempts")
            return "[]"
        # print(responses.text)
        try:
            return responses.text
        except ValueError:
            raise Exception(responses.text)
        
    else: # HuggingFace
        _chat_history = list()
        _chat_history.append({"role": "system", "content": system_instruction})
        for ch in chat_history:
            if 'content' not in ch:
                ch['content'] = ch['text']
            _chat_history.append({"role": ch["role"], "content": ch["content"]})
        _chat_history.append({"role": "user", "content": prompt})

    if not pipeline:
        pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto", # cuda
        framework="pt"
    )
        
    outputs = pipeline(
        _chat_history,
        temperature=0.0001,
        max_new_tokens=128,  # todo, check me
        do_sample=True
     )
    
    return outputs[0]["generated_text"][-1]['content']

    # raise RuntimeError(f"Unknown model: {model}")
    # else:
    #     raise RuntimeError(f"Unknown model: {model}")
        # for chunk in responses:
        #     print(chunk)
        #     text_response.append(chunk.text)
        # print("".join(text_response))


if __name__ == '__main__':
    # ch_0 = {'role': 'user', 'text': 'You Count apples.'}
    system_instruction = "You count the apples and print an integer. Only that, please."
    ch_1 = {'role': 'user', 'text': 'You know that George has one apple.'}
    ch_2 = {'role': 'model', 'text': 'George has one apple.'}
    ch_3 = {'role': 'user', 'text': 'You know that Mary has two apples.'}

    chat_history = [ch_1, ch_2]

    r = execute_prompt(model="gemini-1.5-flash-001", chat_history=chat_history, system_instruction=system_instruction,
                       prompt=ch_3['text'])
    print(r)
