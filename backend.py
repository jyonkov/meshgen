import ast
import contextlib
import io
import queue
import threading
import traceback

import bpy
from smolagents import CodeAgent, HfApiModel, LiteLLMModel, LogLevel

from .tools import ToolManager
from .utils import get_available_models, get_models_dir


class _FinalAnswer(Exception):
    def __init__(self, value):
        self.value = value


class UnhingedExecutor:
    """Raw exec() executor — no AST checks, no import filter, no operator gating."""

    def __init__(self):
        self.state = {}
        self.static_tools = {}
        self.custom_tools = {}

    def send_variables(self, variables):
        self.state.update(variables)

    def send_tools(self, tools):
        self.static_tools = dict(tools)

    def __call__(self, code_action):
        def _final_answer(value=None):
            raise _FinalAnswer(value)

        namespace = {
            **self.state,
            **self.static_tools,
            "final_answer": _final_answer,
            "__builtins__": __builtins__,
        }

        tree = ast.parse(code_action, mode="exec")
        last_expr = (
            tree.body.pop()
            if (tree.body and isinstance(tree.body[-1], ast.Expr))
            else None
        )

        buf = io.StringIO()
        output, is_final_answer = None, False
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, "<agent>", "exec"), namespace)
                if last_expr is not None:
                    output = eval(
                        compile(ast.Expression(last_expr.value), "<agent>", "eval"),
                        namespace,
                    )
        except _FinalAnswer as fa:
            output = fa.value
            is_final_answer = True

        for k, v in namespace.items():
            if k == "__builtins__" or k == "final_answer" or k in self.static_tools:
                continue
            self.state[k] = v

        return output, buf.getvalue(), is_final_answer


class Backend:
    """Singleton class that manages AI model loading and inference."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(Backend, cls).__new__(cls)
                cls._instance.model = None
                cls._instance.agent = None
        return cls._instance

    def is_valid(self):
        prefs = bpy.context.preferences.addons[__package__].preferences
        if prefs.backend_type == "LOCAL":
            return prefs.current_model in get_available_models()
        elif prefs.backend_type == "REMOTE":
            return (
                (
                    prefs.llm_provider == "ollama"
                    and prefs.ollama_endpoint
                    and prefs.ollama_model_name
                )
                or (
                    prefs.llm_provider == "huggingface"
                    and prefs.huggingface_model_id
                    and prefs.huggingface_api_key
                )
                or (
                    prefs.llm_provider == "anthropic"
                    and prefs.anthropic_model_id
                    and prefs.anthropic_api_key
                )
                or (
                    prefs.llm_provider == "openai"
                    and prefs.openai_model_id
                    and prefs.openai_api_key
                )
                or (
                    prefs.llm_provider == "deepseek"
                    and prefs.deepseek_model_id
                    and prefs.deepseek_api_key
                )
            )

    def is_loaded(self):
        return self.model is not None and self.agent is not None

    def _load_local_model(self):
        from .utils import LlamaCppModel

        prefs = bpy.context.preferences.addons[__package__].preferences
        model_path = get_models_dir() / prefs.current_model
        self.model = LlamaCppModel(
            model_path=str(model_path),
            n_gpu_layers=-1,
            n_ctx=prefs.context_length,
            max_tokens=prefs.context_length,
        )

    def _load_hf_api_model(self):
        print("Loading Hugging Face API model")
        prefs = bpy.context.preferences.addons[__package__].preferences
        model_id = prefs.huggingface_model_id
        token = prefs.huggingface_api_key
        self.model = HfApiModel(
            model_id=model_id,
            token=token,
            max_tokens=prefs.context_length,
        )

    def _load_litellm_model(self):
        prefs = bpy.context.preferences.addons[__package__].preferences
        kwargs = {}
        if prefs.llm_provider == "ollama":
            model_id = f"ollama_chat/{prefs.ollama_model_name}"
            api_base = prefs.ollama_endpoint
            api_key = prefs.ollama_api_key or None
            kwargs["num_ctx"] = prefs.context_length
        elif prefs.llm_provider == "anthropic":
            model_id = f"anthropic/{prefs.anthropic_model_id}"
            api_base = None
            api_key = prefs.anthropic_api_key
        elif prefs.llm_provider == "openai":
            model_id = prefs.openai_model_id
            api_base = None
            api_key = prefs.openai_api_key
        elif prefs.llm_provider == "deepseek":
            model_id = f"deepseek/{prefs.deepseek_model_id}"
            api_base = None
            api_key = prefs.deepseek_api_key
        else:
            raise ValueError(f"Unknown provider: {prefs.llm_provider}")

        self.model = LiteLLMModel(
            model_id=model_id,
            api_base=api_base,
            api_key=api_key,
            **kwargs,
        )
        try:
            input_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello, world!",
                        }
                    ],
                }
            ]
            self.model(input_messages)
        except Exception as e:
            self.model = None
            raise e

    def load(self):
        prefs = bpy.context.preferences.addons[__package__].preferences
        if prefs.backend_type == "LOCAL":
            self._load_local_model()
        elif prefs.backend_type == "REMOTE":
            if prefs.llm_provider == "huggingface":
                self._load_hf_api_model()
            elif (
                prefs.llm_provider == "ollama"
                or prefs.llm_provider == "anthropic"
                or prefs.llm_provider == "openai"
                or prefs.llm_provider == "deepseek"
            ):
                self._load_litellm_model()
            else:
                raise ValueError(f"Unknown provider: {prefs.llm_provider}")
        else:
            raise ValueError("Invalid backend type")

        self.agent = CodeAgent(
            model=self.model,
            tools=ToolManager.instance().tools,
            additional_authorized_imports=["*"],
            add_base_tools=False,
            verbosity_level=LogLevel.DEBUG,
        )
        self.agent.python_executor = UnhingedExecutor()

    def start_chat_completion(self, messages, temperature, stop_event):
        prompt = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        self.model.temperature = temperature
        output_queue = queue.Queue()

        def run_agent():
            try:
                for step in self.agent.run(prompt, stream=True):
                    if stop_event.is_set():
                        output_queue.put(("CANCELED", None))
                        break
                    self._step_callback(step, output_queue)
            except Exception as e:
                output_queue.put(("ERROR", (str(e), traceback.format_exc())))
            finally:
                if not stop_event.is_set():
                    output_queue.put(("DONE", None))

        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()
        return output_queue

    def _step_callback(self, step, output_queue):
        if hasattr(step, "error") and step.error:
            output_queue.put(("STEP_ERROR", step.error.message))
        elif hasattr(step, "model_output"):
            thought = step.model_output.strip() if step.model_output else "None"
            action = (
                str([tc.dict() for tc in step.tool_calls])
                if step.tool_calls
                else "None"
            )
            observation = step.observations if step.observations else "None"
            full_output = (
                f"Step {step.step_number}\n\n"
                f"{thought}\n"
                f"\n"
                f"Action:\n"
                f"{action}\n"
                f"\n"
                f"Observation:\n"
                f"{observation}\n"
            )
            output_queue.put(("STEP", (str(thought), str(full_output))))
            if step.tool_calls and "final_answer" in step.tool_calls[0].arguments:
                output_queue.put(("FINAL_ANSWER", str(step.action_output)))

    @classmethod
    def instance(cls):
        return cls()

    @classmethod
    def reset(cls):
        with cls._lock:
            if cls._instance:
                if hasattr(cls._instance, "model") and cls._instance.model:
                    try:
                        del cls._instance.model
                    except Exception:
                        pass

                if hasattr(cls._instance, "agent") and cls._instance.agent:
                    try:
                        del cls._instance.agent
                    except Exception:
                        pass

                cls._instance = None

        ToolManager.reset()
