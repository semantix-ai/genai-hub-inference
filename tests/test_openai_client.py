import asyncio
from semantix_genai_inference import ModelClient

class OpenAIInferenceClientTests:

    def test_generate_async_with_streaming(self):
        client = ModelClient.create("openai")
        prompt = [{"role": "user", "content": "Once upon a time"}]
        temperature = 0.5
        top_p = 1.0
        n = 1
        presence_penalty = 0.0
        frequency_penalty = 0.0
        async def test():
            async for line in client.generate_stream(prompt, temperature=temperature, top_p=top_p, n=n,
                                                presence_penalty=presence_penalty, frequency_penalty=frequency_penalty):
                print(line)
        asyncio.run(test())

if __name__ == "__main__":
    OpenAIInferenceClientTests().test_generate_async_with_streaming()
    print("Done!")