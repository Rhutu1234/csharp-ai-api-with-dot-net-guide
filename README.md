# AI APIs with .NET

*A deep-dive walkthrough of calling hosted AI models from .NET — covering OpenAI API integration with the official .NET library, Azure OpenAI and how it differs, the chat/completions model and why conversations are stateless, streaming responses to users, function/tool calling, structured (schema-constrained) outputs, token management and context windows, temperature and other model parameters, error handling and retries, client-side rate limiting, and the practical levers for keeping API costs under control.*

---

## Table of Contents

1. [Introduction](#introduction)
2. [The Landscape and Setup](#1-the-landscape-and-setup)
3. [OpenAI API Integration](#2-openai-api-integration)
4. [Azure OpenAI](#3-azure-openai)
5. [Chat / Completions APIs](#4-chat--completions-apis)
6. [Streaming Responses](#5-streaming-responses)
7. [Function / Tool Calling](#6-function--tool-calling)
8. [Structured Outputs](#7-structured-outputs)
9. [Token Management](#8-token-management)
10. [Temperature and Model Parameters](#9-temperature-and-model-parameters)
11. [Error Handling and Retries](#10-error-handling-and-retries)
12. [Rate Limiting](#11-rate-limiting)
13. [API Cost Optimization](#12-api-cost-optimization)
14. [A Provider-Neutral Option: Microsoft.Extensions.AI](#13-a-provider-neutral-option-microsoftextensionsai)
15. [Common Pitfalls](#14-common-pitfalls)
16. [Quick Reference Table](#quick-reference-table)
17. [Conclusion](#conclusion)

---

## Introduction

Calling a large language model from .NET looks deceptively simple — a few lines of code and text comes back. The difficulty is everything around those few lines: the model has no memory between calls, responses can take many seconds, the output is free text unless you constrain it, every call costs money in proportion to the text sent and received, and the service will rate-limit and occasionally fail. Production-quality AI integration is mostly about handling those realities well.

```csharp
using OpenAI.Chat;

ChatClient client = new(model: "gpt-4o-mini", apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY"));

ChatCompletion completion = await client.CompleteChatAsync("Explain dependency injection in one sentence.");
Console.WriteLine(completion.Content[0].Text);
```

That is the entire "hello world." This guide builds from it toward something you could run in production: conversation history, streaming, tools, validated JSON output, token budgeting, retries, rate limits, and cost controls.

> **A note on model names and API versions.** Model names in this guide (`gpt-4o-mini` and similar) are placeholders — provider model lineups change frequently. Keep the model name in configuration, not in code, and check the provider's current model list and pricing page. Likewise, SDK property names have shifted between package versions (for example, `MaxOutputTokenCount` was previously named differently), so verify against the version you install.

---

## 1. The Landscape and Setup

### Two routes to the same family of models

```plaintext
OpenAI API           -> api.openai.com. Fastest access to new models and features.
                        Authenticated with an API key. Billed by OpenAI.

Azure OpenAI         -> OpenAI-family models hosted in YOUR Azure subscription.
                        Adds Azure's identity (Entra ID), private networking,
                        regional deployment, quotas, content filtering, and
                        enterprise compliance. Billed through Azure.
```

Both are consumed from .NET through the same client-library design, which is why most code in this guide works against either with only the client construction changing.

### Packages

```plaintext
dotnet add package OpenAI                        # official OpenAI .NET library
dotnet add package Azure.AI.OpenAI               # Azure OpenAI (builds on the OpenAI library)
dotnet add package Azure.Identity                # DefaultAzureCredential / Entra ID auth
dotnet add package Microsoft.ML.Tokenizers       # count tokens before sending (Section 8)
dotnet add package Microsoft.Extensions.AI.OpenAI  # provider-neutral IChatClient (Section 13)
```

### Choosing between them

```plaintext
Choose OpenAI directly when:  you're prototyping, want newest models first,
                              or have no Azure footprint.
Choose Azure OpenAI when:     your organization already runs on Azure, needs
                              data-residency / private-endpoint / Entra ID
                              (keyless) auth, or requires compliance controls
                              and consolidated billing.
```

### Secrets: the one non-negotiable

```plaintext
NEVER hard-code an API key, commit it to source control, or ship it in a client
(mobile app, SPA, desktop binary). A key in client code is a key anyone can extract
and spend against your account.

Development:  dotnet user-secrets, or environment variables
Production:   Azure Key Vault / your platform's secret store, or — on Azure —
              keyless Entra ID authentication (Section 3)
Architecture: clients call YOUR backend; only the backend talks to the AI provider
```

---

## 2. OpenAI API Integration

### Create the client once and reuse it

```csharp
// Program.cs (ASP.NET Core)
builder.Services.AddSingleton(sp =>
{
    var config = sp.GetRequiredService<IConfiguration>();
    return new ChatClient(
        model:  config["OpenAI:Model"],          // e.g. "gpt-4o-mini" — configurable, not hard-coded
        apiKey: config["OpenAI:ApiKey"]);        // from user-secrets / Key Vault
});
```

Client objects are designed to be thread-safe and reused: register `ChatClient` as a **singleton**. Constructing a new client per request wastes connection setup and defeats connection reuse.

### Using it from an endpoint

```csharp
app.MapPost("/summarize", async (SummarizeRequest req, ChatClient chat, CancellationToken ct) =>
{
    List<ChatMessage> messages =
    [
        new SystemChatMessage("You summarize text in two sentences."),
        new UserChatMessage(req.Text),
    ];

    ChatCompletion completion = await chat.CompleteChatAsync(messages, cancellationToken: ct);
    return Results.Ok(new { summary = completion.Content[0].Text });
});

public record SummarizeRequest(string Text);
```

Two habits worth forming immediately:

```plaintext
- Pass the CancellationToken through. If the caller disconnects, the request stops
  instead of continuing to wait on (and be billed for) a response nobody will read.
- Cap what callers can send. An endpoint that forwards arbitrary-length user text to
  a paid model is an invitation to run up your bill (Sections 8, 11, 12).
```

### The two OpenAI API styles

```plaintext
Chat Completions : the established, widely supported request/response format —
                   a list of role-tagged messages in, an assistant message out.
                   This guide focuses on it. Azure OpenAI supports it too.
Responses API    : OpenAI's newer API, with built-in conversation state and
                   built-in tools; the .NET library exposes it via a separate client.
```

The concepts here — messages, tokens, tools, streaming, structured output — carry over; consult current documentation to decide which API suits a new project.

---

## 3. Azure OpenAI

### Same client types, different construction

```csharp
using Azure.AI.OpenAI;
using Azure.Identity;
using OpenAI.Chat;

// Option A: API key
AzureOpenAIClient azureClient = new(
    new Uri("https://my-resource.openai.azure.com/"),
    new System.ClientModel.ApiKeyCredential(apiKey));

// Option B (preferred in production): keyless, via Microsoft Entra ID
AzureOpenAIClient azureClient2 = new(
    new Uri("https://my-resource.openai.azure.com/"),
    new DefaultAzureCredential());

// Ask for a chat client by DEPLOYMENT name — then everything else is identical
ChatClient chat = azureClient2.GetChatClient("my-gpt-deployment");
ChatCompletion completion = await chat.CompleteChatAsync("Hello!");
```

After `GetChatClient`, the rest of the code in this guide — messages, streaming, tools, structured outputs — works unchanged. That is the payoff of the shared client design.

### What is genuinely different on Azure

```plaintext
DEPLOYMENT NAMES, not model names.
  In Azure you first DEPLOY a model under a name you choose ("my-gpt-deployment"),
  and your code passes that deployment name where OpenAI's API takes a model name.
  A "404 not found" often means a wrong deployment name, or the deployment is in a
  different resource/region than the endpoint you're calling.

KEYLESS AUTH via Entra ID.
  DefaultAzureCredential uses your developer login locally and a managed identity
  in Azure — no secrets to store or rotate. Assign the identity an appropriate
  role on the Azure OpenAI resource (e.g. "Cognitive Services OpenAI User").

QUOTAS PER DEPLOYMENT.
  Rate limits (tokens/minute, requests/minute) are assigned per deployment and
  region, and you manage them in Azure. Hitting them returns 429 (Section 11).

CONTENT FILTERING.
  Azure applies content-safety filters by default. A filtered prompt can return an
  error; a filtered response can end with a content-filter finish reason. Handle
  both rather than assuming every call returns normal text.

NETWORKING AND COMPLIANCE.
  Private endpoints, virtual networks, regional data processing, and enterprise
  compliance certifications are the main reasons organizations choose Azure.

MODEL AVAILABILITY.
  New models and features can reach Azure after the OpenAI API, and availability
  varies by region — check what your region offers before designing around a model.
```

---

## 4. Chat / Completions APIs

### Messages in, an assistant message out

A chat request is a list of messages, each with a **role**:

```plaintext
system     -> instructions defining behavior, tone, and rules ("You are a support
              assistant for ACME. Answer only from the provided context.")
user       -> what the end user said
assistant  -> what the model said previously (history you replay back to it)
tool       -> the result of a function the model asked you to run (Section 6)
```

```csharp
List<ChatMessage> messages =
[
    new SystemChatMessage("You are a concise C# tutor."),
    new UserChatMessage("What is a record?"),
];

ChatCompletion first = await client.CompleteChatAsync(messages);
messages.Add(new AssistantChatMessage(first));                     // replay the model's own answer
messages.Add(new UserChatMessage("How is it different from a class?"));

ChatCompletion second = await client.CompleteChatAsync(messages);  // now it has the context
```

### The model has no memory — YOU hold the conversation

```plaintext
Every call is STATELESS. The model only knows what is in the messages you send
THIS time. "Remembering" the conversation means re-sending the history with every
request. Consequences:

  - You must store conversation history (in memory, a cache, or a database).
  - Every extra turn makes the NEXT request bigger — and more expensive.
  - History eventually exceeds the context window and must be trimmed (Section 8).
```

### Reading the response

```csharp
string text = completion.Content[0].Text;
ChatFinishReason reason = completion.FinishReason;
ChatTokenUsage usage = completion.Usage;     // InputTokenCount, OutputTokenCount, TotalTokenCount
```

**Always check `FinishReason`** — it tells you *why* generation stopped:

```plaintext
Stop           -> the model finished naturally. The normal case.
Length         -> it hit the maximum output tokens (or the context limit) and was
                  TRUNCATED mid-answer. Truncated JSON in particular will not parse.
ToolCalls      -> the model wants you to run one or more functions (Section 6).
ContentFilter  -> output was withheld by a content filter.
```

### Prompting structure that holds up in production

```plaintext
- Put stable instructions in the SYSTEM message; put per-request data in the USER message.
- Clearly delimit untrusted content (a user's document, a web page) from your instructions.
  Text inside it can attempt "prompt injection" — instructions disguised as data.
  Never let model output you haven't validated trigger privileged actions.
- Be specific about format and length. "Answer in at most three bullet points" is
  both better UX and cheaper (fewer output tokens).
```

---

## 5. Streaming Responses

### Show text as it's generated instead of waiting for the whole answer

A long response can take many seconds to complete. Streaming returns it in small pieces as they're produced, so users see the first words almost immediately — a large improvement in *perceived* speed.

```csharp
await foreach (StreamingChatCompletionUpdate update in client.CompleteChatStreamingAsync(messages))
{
    foreach (ChatMessageContentPart part in update.ContentUpdate)
        Console.Write(part.Text);
}
```

### Streaming from an ASP.NET Core endpoint (Server-Sent Events)

```csharp
app.MapPost("/chat/stream", async (ChatRequest req, ChatClient client, HttpContext ctx, CancellationToken ct) =>
{
    ctx.Response.ContentType = "text/event-stream";
    ctx.Response.Headers.CacheControl = "no-cache";

    List<ChatMessage> messages = [ new SystemChatMessage("You are helpful."), new UserChatMessage(req.Message) ];

    await foreach (var update in client.CompleteChatStreamingAsync(messages, cancellationToken: ct))
    {
        foreach (var part in update.ContentUpdate)
        {
            // JSON-encode each fragment so newlines and special characters can't break the SSE framing
            await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(part.Text)}\n\n", ct);
            await ctx.Response.Body.FlushAsync(ct);
        }
    }

    await ctx.Response.WriteAsync("data: [DONE]\n\n", ct);
});

public record ChatRequest(string Message);
```

### Things specific to streaming

```plaintext
- Flush after each chunk, or the server may buffer and the user sees nothing until the end.
- Pass the CancellationToken. If the user closes the tab, stop consuming the stream
  so you aren't generating and paying for output no one will read.
- ERRORS CAN HAPPEN MID-STREAM, after the HTTP 200 and some text have already been
  sent. You can't change the status code any more — send an error event the client
  understands, and design the UI to cope with a partial answer.
- Streamed TOOL CALLS arrive as FRAGMENTS (name and JSON arguments split across
  updates). You must concatenate the pieces before parsing the arguments (Section 6).
- Usage/token counts and content-filter signals arrive differently when streaming;
  check the documentation for your library version if you rely on them for billing.
- Streaming improves responsiveness, NOT cost: you pay the same tokens either way.
```

---

## 6. Function / Tool Calling

### Letting the model ask your code to do things

A model can't check the weather, query your database, or place an order. **Tool calling** lets you describe functions to the model; when it decides one is needed, it replies with a *request to call it* — the function name and JSON arguments — instead of text. **Your code runs the function** and sends the result back; the model then writes its final answer.

```plaintext
1. You send:   user question  +  tool DEFINITIONS (name, description, JSON schema)
2. Model says: "call get_weather with {"city": "Pune"}"       (FinishReason = ToolCalls)
3. YOU run get_weather("Pune") in your own code
4. You send:   the whole conversation + the tool's RESULT
5. Model says: the final natural-language answer
```

### Defining a tool and running the loop

```csharp
ChatTool weatherTool = ChatTool.CreateFunctionTool(
    functionName: "get_weather",
    functionDescription: "Get the current weather for a city.",
    functionParameters: BinaryData.FromString("""
    {
      "type": "object",
      "properties": {
        "city": { "type": "string", "description": "City name, e.g. Pune" }
      },
      "required": ["city"]
    }
    """));

var options = new ChatCompletionOptions { Tools = { weatherTool } };

List<ChatMessage> messages = [ new UserChatMessage("Do I need an umbrella in Pune today?") ];

bool done = false;
int rounds = 0;
while (!done && rounds++ < 5)                                  // hard cap: never loop forever
{
    ChatCompletion completion = await client.CompleteChatAsync(messages, options);

    switch (completion.FinishReason)
    {
        case ChatFinishReason.ToolCalls:
            messages.Add(new AssistantChatMessage(completion));     // the model's tool request MUST be in history

            foreach (ChatToolCall call in completion.ToolCalls)
            {
                string result;
                switch (call.FunctionName)
                {
                    case "get_weather":
                        using (JsonDocument args = JsonDocument.Parse(call.FunctionArguments))
                        {
                            string city = args.RootElement.GetProperty("city").GetString()!;
                            result = await weatherService.GetSummaryAsync(city);
                        }
                        break;
                    default:
                        result = $"Error: unknown tool '{call.FunctionName}'.";
                        break;
                }
                messages.Add(new ToolChatMessage(call.Id, result));  // match the result to the call's Id
            }
            break;

        case ChatFinishReason.Stop:
            Console.WriteLine(completion.Content[0].Text);
            done = true;
            break;

        default:
            throw new InvalidOperationException($"Unexpected finish reason: {completion.FinishReason}");
    }
}
```

### Rules that make tool calling safe and reliable

```plaintext
- The MODEL NEVER EXECUTES ANYTHING. It only proposes calls; your code decides
  whether to run them. That is your security boundary.
- VALIDATE the arguments. They come from a model, steered by user input. Treat them
  exactly like untrusted user input: parse defensively, check ranges and allowed values,
  and never concatenate them into SQL or shell commands (use parameters — see this
  series' SQL guides).
- Authorize on the SERVER, as the real user. "The model asked for it" is not authorization.
  A tool that deletes data or spends money should require confirmation or a permission check.
- Make tool results SMALL and relevant. Everything returned goes back into the
  model's context and counts as input tokens.
- Return errors as TEXT to the model ("City not found") rather than throwing, so it
  can recover or apologize gracefully.
- Handle MULTIPLE tool calls in one response (the model may request several at once),
  and cap the number of rounds as shown above.
- The tool DESCRIPTION is what the model reads to decide when to use it — write it
  like documentation for a new colleague. Vague descriptions cause wrong or missed calls.
```

---

## 7. Structured Outputs

### Getting JSON you can actually deserialize

Asking a model "reply in JSON" works most of the time — and "most of the time" fails in production when a stray sentence or a missing field breaks your parser. **Structured outputs** constrain the model to produce JSON that conforms to a JSON Schema you supply.

```csharp
public record Invoice(string Vendor, string InvoiceNumber, decimal Total, string Currency);

var options = new ChatCompletionOptions
{
    ResponseFormat = ChatResponseFormat.CreateJsonSchemaFormat(
        jsonSchemaFormatName: "invoice",
        jsonSchema: BinaryData.FromString("""
        {
          "type": "object",
          "properties": {
            "Vendor":        { "type": "string" },
            "InvoiceNumber": { "type": "string" },
            "Total":         { "type": "number" },
            "Currency":      { "type": "string" }
          },
          "required": ["Vendor", "InvoiceNumber", "Total", "Currency"],
          "additionalProperties": false
        }
        """),
        jsonSchemaIsStrict: true)
};

ChatCompletion completion = await client.CompleteChatAsync(
    [ new SystemChatMessage("Extract the invoice fields."), new UserChatMessage(invoiceText) ],
    options);

Invoice? invoice = JsonSerializer.Deserialize<Invoice>(completion.Content[0].Text);
```

### How strict mode behaves

```plaintext
With strict mode on:
  - The output is constrained to match the schema.
  - The schema must follow the supported subset: every property listed in
    "required", and "additionalProperties": false on objects.
  - Optional fields are expressed as nullable types rather than omitted properties.
  - Not every JSON Schema feature is supported — consult the current docs.
```

### Still validate — the schema guarantees SHAPE, not TRUTH

```plaintext
A structured output is guaranteed to be well-formed JSON of the right shape. It is NOT
guaranteed to be CORRECT. The model can return a perfectly valid Invoice with a
hallucinated total. Therefore:

  - Apply business validation (is Total positive? is Currency a real ISO code?).
  - Check completion.FinishReason — a Length stop can still truncate the output.
  - Check for a REFUSAL: when the model declines a request for safety reasons, the
    response carries a refusal message instead of schema-conforming JSON, so inspect
    completion.Refusal before deserializing.
  - Use try/catch around Deserialize anyway; defensive code costs nothing.
```

### Structured outputs vs. tool calling

```plaintext
Use STRUCTURED OUTPUT when you want the model's FINAL ANSWER in a fixed shape
  (extraction, classification, form-filling).
Use TOOL CALLING when the model needs to TRIGGER ACTIONS or fetch data mid-conversation.
```

---

## 8. Token Management

### Tokens are the unit of everything: cost, speed, and limits

A **token** is a chunk of text — roughly four characters, or about three-quarters of an English word, though it varies by language (many non-English languages and code use more tokens per word). The model reads and writes tokens, and you are billed per token, with input and output tokens usually priced differently (output is typically more expensive).

```plaintext
Context window = INPUT tokens + OUTPUT tokens, combined, must fit within the model's limit.

  Your request:  system prompt + full chat history + tool definitions + new user message
  = INPUT tokens
  + the reply the model generates
  = OUTPUT tokens
```

### Measure actual usage from every response

```csharp
ChatTokenUsage usage = completion.Usage;
logger.LogInformation("Tokens in={In} out={Out} total={Total}",
    usage.InputTokenCount, usage.OutputTokenCount, usage.TotalTokenCount);
```

Log these on every call. It is the foundation of cost tracking (Section 12) and of noticing when a prompt change has quietly doubled your spend.

### Count tokens *before* sending

```csharp
using Microsoft.ML.Tokenizers;

Tokenizer tokenizer = TiktokenTokenizer.CreateForModel("gpt-4o");
int tokenCount = tokenizer.CountTokens(userText);

if (tokenCount > 4_000)
    return Results.BadRequest("Input too long.");
```

Counts from a local tokenizer are estimates for chat requests — the message framing adds a few tokens per message — so leave headroom rather than packing right up to the limit.

### Keep conversation history within budget

Because history is re-sent on every turn, long conversations get progressively slower and costlier — and eventually exceed the window. Store history in your own type so you can trim it deliberately:

```csharp
public record Turn(string Role, string Text);   // "user" or "assistant"

static List<Turn> TrimToBudget(List<Turn> history, Tokenizer tokenizer, int maxHistoryTokens)
{
    var kept = new List<Turn>();
    int used = 0;

    // Walk BACKWARD from the newest turn, keeping as many recent turns as fit
    for (int i = history.Count - 1; i >= 0; i--)
    {
        int cost = tokenizer.CountTokens(history[i].Text) + 4;     // +4: rough per-message overhead
        if (used + cost > maxHistoryTokens) break;
        kept.Insert(0, history[i]);
        used += cost;
    }
    return kept;
}
```

```csharp
// Build the request: the system prompt is ALWAYS included; only the history is trimmed
var messages = new List<ChatMessage> { new SystemChatMessage(systemPrompt) };
foreach (var turn in TrimToBudget(history, tokenizer, maxHistoryTokens: 3_000))
    messages.Add(turn.Role == "user" ? new UserChatMessage(turn.Text) : new AssistantChatMessage(turn.Text));
```

### Strategies when conversations get long

```plaintext
Sliding window      : keep the last N turns (simple, loses early context).
Summarization       : periodically ask the model to summarize older turns into a short
                      note, then keep the summary + recent turns. Costs an extra call
                      but preserves the gist.
Retrieval (RAG)     : don't stuff everything into the prompt; store documents/facts
                      and retrieve only the few relevant pieces per question
                      (see this series' AI/ML Fundamentals guide on embeddings).
Cap OUTPUT size too : set a maximum output token count appropriate to the task (Section 9).
```

---

## 9. Temperature and Model Parameters

### Settings that shape the model's output

```csharp
var options = new ChatCompletionOptions
{
    Temperature = 0.2f,
    TopP = 1.0f,
    MaxOutputTokenCount = 500,
    FrequencyPenalty = 0.0f,
    PresencePenalty = 0.0f,
};
options.StopSequences.Add("END_OF_ANSWER");

ChatCompletion completion = await client.CompleteChatAsync(messages, options);
```

### What each parameter does

```plaintext
Temperature (commonly 0 to 2)
  Controls randomness in word choice.
    LOW  (0 - 0.3)  -> focused and more consistent. Extraction, classification, code, Q&A.
    MID  (0.5 - 0.8)-> balanced.
    HIGH (1.0+)     -> more varied and surprising. Brainstorming, creative writing.
  Low temperature makes output MORE REPEATABLE, but NOT guaranteed identical:
  results can still vary between calls.

Top-p (nucleus sampling)
  Restricts choices to the smallest set of likely tokens whose probabilities add up to p.
  An alternative way to control randomness. Adjust temperature OR top-p, not both at once —
  changing both makes behavior hard to reason about.

Max output tokens
  A hard cap on the reply length. Your main guard against runaway (expensive) responses —
  but set it high enough: too low and answers are cut off (FinishReason = Length).

Frequency / presence penalty
  Discourage repeating the same words (frequency) or revisiting the same topics (presence).
  Useful occasionally for repetitive output; leave at defaults otherwise.

Stop sequences
  Strings that, when generated, end the response immediately. Handy for fixed formats.
```

### Choosing parameters by task

```plaintext
Data extraction / classification / JSON output   : low temperature, structured output
Customer-support answers grounded in documents   : low temperature
Code generation                                  : low temperature
Marketing copy / ideation                        : higher temperature
```

### Two cautions

```plaintext
- Some models — notably "reasoning" models — restrict or ignore sampling parameters like
  temperature, and use different controls (such as a reasoning-effort setting) and a
  different name for the output cap. If a parameter is rejected with a 400 error, check
  that model's documentation before assuming your code is wrong.
- Tune parameters against a REPRESENTATIVE test set of prompts, not one example.
  Anecdotes mislead; a small evaluation set tells you whether a change actually helped.
```

---

## 10. Error Handling and Retries

### The service will sometimes say no — plan for it

```csharp
using System.ClientModel;

try
{
    ChatCompletion completion = await client.CompleteChatAsync(messages, options, ct);
    // ...
}
catch (ClientResultException ex)
{
    // ex.Status holds the HTTP status code
    logger.LogError(ex, "AI call failed with HTTP {Status}", ex.Status);
    // decide what to do based on the status — table below
}
catch (OperationCanceledException) when (ct.IsCancellationRequested)
{
    // The caller cancelled; not an error worth alerting on
}
```

### What each status means — and whether retrying helps

```plaintext
400  Bad request       -> A bug in YOUR request: malformed message list, unsupported
                          parameter for this model, or input exceeds the context window.
                          DO NOT RETRY. Fix the request (or trim the input).
401  Unauthorized      -> Bad/expired/missing credentials. DO NOT RETRY. Fix the key/identity.
403  Forbidden         -> No permission/role, or blocked region/policy. DO NOT RETRY.
404  Not found         -> Wrong model name or (Azure) deployment name / endpoint. DO NOT RETRY.
408  Timeout           -> RETRY with backoff.
429  Too many requests -> Either you're RATE-LIMITED (RETRY after waiting) or you've run out
                          of QUOTA/BILLING credit (retrying will NOT help — inspect the
                          error message to tell which).
500/502/503/504        -> Provider-side trouble. RETRY with backoff.
```

### The library already retries transient failures

```plaintext
The OpenAI .NET library retries certain transient errors (e.g. 408, 429, and 5xx)
automatically a small number of times with exponential backoff. You can tune this and
the overall timeout through client options:
```

```csharp
var clientOptions = new OpenAIClientOptions
{
    RetryPolicy = new ClientRetryPolicy(maxRetries: 5),
    NetworkTimeout = TimeSpan.FromSeconds(60),
};

ChatClient client = new("gpt-4o-mini", new ApiKeyCredential(apiKey), clientOptions);
```

### When you need more: your own retry with jittered backoff

```csharp
static async Task<T> WithRetryAsync<T>(Func<Task<T>> action, int maxAttempts = 4, CancellationToken ct = default)
{
    for (int attempt = 1; ; attempt++)
    {
        try
        {
            return await action();
        }
        catch (ClientResultException ex) when (IsTransient(ex.Status) && attempt < maxAttempts)
        {
            // Honor the server's Retry-After hint if it sent one
            TimeSpan delay = TimeSpan.Zero;
            if (ex.GetRawResponse()?.Headers.TryGetValue("Retry-After", out var value) == true
                && int.TryParse(value, out int seconds))
                delay = TimeSpan.FromSeconds(seconds);

            if (delay == TimeSpan.Zero)
            {
                // Exponential backoff (1s, 2s, 4s...) plus RANDOM JITTER
                double baseSeconds = Math.Pow(2, attempt - 1);
                delay = TimeSpan.FromSeconds(baseSeconds + Random.Shared.NextDouble());
            }

            await Task.Delay(delay, ct);
        }
    }
}

static bool IsTransient(int status) => status is 408 or 429 or >= 500;
```

```plaintext
Why JITTER matters: if 1,000 clients all back off for exactly 2 seconds, they ALL
retry at the same instant and overload the service again. Random jitter spreads the
retries out.
```

### Resilience beyond retries

```plaintext
- TIMEOUTS: always set one. A hung call that ties up a request thread is worse than a failure.
- CIRCUIT BREAKER: when the provider is clearly down, stop hammering it for a while and fail
  fast (libraries such as Polly / Microsoft.Extensions.Resilience provide this).
- FALLBACKS: degrade gracefully — a cached answer, a cheaper model, a simpler non-AI
  response, or an honest "this feature is temporarily unavailable."
- IDEMPOTENCY: a retried request may be processed twice. If a tool call has a side effect
  (send an email, charge a card), guard it with an idempotency key so a retry can't repeat it.
- NEVER log secrets, and think carefully before logging full prompts and responses —
  they may contain personal or confidential data.
```

---

## 11. Rate Limiting

### Providers limit how fast you can go — in two dimensions

```plaintext
RPM  requests per minute      TPM  tokens per minute (input + output)

Exceeding either returns HTTP 429. Limits depend on your account tier (OpenAI) or your
deployment's configured quota (Azure OpenAI), and they apply per model/deployment.
```

Because **TPM** counts tokens, a handful of huge prompts can exhaust your limit just as surely as thousands of tiny ones. Rate-limiting by request count alone isn't enough.

### Throttle on the client side instead of just reacting to 429s

```csharp
using System.Threading.RateLimiting;

// Allow ~10 calls/second, refilling continuously, with a bounded waiting queue
var limiter = new TokenBucketRateLimiter(new TokenBucketRateLimiterOptions
{
    TokenLimit = 10,
    TokensPerPeriod = 10,
    ReplenishmentPeriod = TimeSpan.FromSeconds(1),
    QueueLimit = 100,
    QueueProcessingOrder = QueueProcessingOrder.OldestFirst,
    AutoReplenishment = true,
});

async Task<ChatCompletion> RateLimitedCallAsync(List<ChatMessage> messages, CancellationToken ct)
{
    using RateLimitLease lease = await limiter.AcquireAsync(permitCount: 1, ct);
    if (!lease.IsAcquired)
        throw new InvalidOperationException("Too many requests queued; try again later.");

    return await client.CompleteChatAsync(messages, cancellationToken: ct);
}
```

### Limit concurrency for batch jobs

```csharp
var gate = new SemaphoreSlim(initialCount: 5);          // at most 5 calls in flight at once

var tasks = documents.Select(async doc =>
{
    await gate.WaitAsync(ct);
    try   { return await SummarizeAsync(doc, ct); }
    finally { gate.Release(); }
});

var summaries = await Task.WhenAll(tasks);
```

Firing `Task.WhenAll` over thousands of items with no gate is a reliable way to trigger a wall of 429 errors.

### Protect your own endpoints too

```csharp
// ASP.NET Core's built-in rate limiter: per-user limits on YOUR API
builder.Services.AddRateLimiter(o =>
{
    o.AddPolicy("ai", httpContext => RateLimitPartition.GetFixedWindowLimiter(
        partitionKey: httpContext.User.Identity?.Name ?? httpContext.Connection.RemoteIpAddress?.ToString() ?? "anon",
        factory: _ => new FixedWindowRateLimiterOptions { PermitLimit = 20, Window = TimeSpan.FromMinutes(1) }));
    o.RejectionStatusCode = StatusCodes.Status429TooManyRequests;
});

app.UseRateLimiter();
app.MapPost("/chat", ChatHandler).RequireRateLimiting("ai");
```

```plaintext
This is both a reliability measure AND a cost-control measure. An unthrottled public AI
endpoint lets one abusive user — or one buggy client in a loop — consume your entire
provider quota and budget, taking the feature down for everyone.
```

### Other practical points

```plaintext
- Responses include rate-limit headers (remaining requests/tokens, reset times) you can
  read to throttle adaptively.
- For non-urgent bulk work, a provider BATCH API (asynchronous, results within hours,
  usually discounted and with separate limits) is often better than hammering the
  real-time endpoint (Section 12).
- Request quota increases (or spread load across deployments/regions on Azure) BEFORE a
  launch, not during one.
```

---

## 12. API Cost Optimization

### Cost = tokens × price — so the levers are tokens and price

Every optimization below reduces either the number of tokens you send/receive, or the price per token you pay.

### 1. Use the smallest model that does the job

```plaintext
Providers offer a spread of models: small/fast/cheap through large/capable/expensive,
with large price gaps between them. Many tasks — classification, extraction, simple
summarization, routing — are handled perfectly well by a small model. Reserve the
expensive model for tasks that genuinely need it.

Pattern: MODEL ROUTING. Try the cheap model first; escalate to the larger one only when
the task is complex or the cheap model's answer fails validation.
```

### 2. Send fewer input tokens

```plaintext
- Trim conversation history (Section 8); don't resend the entire transcript forever.
- Keep the system prompt tight. A 2,000-token system prompt is paid for on EVERY request.
- Use retrieval (RAG): send the 3 relevant paragraphs, not the whole manual.
- Return compact tool results (Section 6) and send only the fields the model needs.
- Define only the tools relevant to the current request — tool definitions are input tokens too.
```

### 3. Cap output tokens

```plaintext
- Set MaxOutputTokenCount appropriate to the task.
- Ask for concise formats ("answer in one sentence", "return JSON only").
- Output tokens usually cost more than input tokens, so verbosity is expensive.
```

### 4. Exploit prompt caching

```plaintext
Providers can discount input tokens when a request begins with a prefix they've recently
processed (OpenAI applies this automatically for sufficiently long, repeated prefixes;
other providers' mechanisms differ). To benefit:
  - Put STATIC content FIRST (system prompt, instructions, reference documents, tool
    definitions) and the VARIABLE content (the user's question) LAST.
  - Keep the static prefix byte-for-byte identical across requests.
Check the provider's documentation for current minimum lengths and discount levels.
```

### 5. Don't call the model when you don't have to

```csharp
// Cache identical, deterministic requests
string key = $"summary:{Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text)))}";

string summary = await cache.GetOrCreateAsync(key, async entry =>
{
    entry.AbsoluteExpirationRelativeToNow = TimeSpan.FromHours(24);
    var completion = await client.CompleteChatAsync($"Summarize in two sentences: {text}");
    return completion.Value.Content[0].Text;
});
```

```plaintext
- Cache responses for repeated, low-temperature requests (IMemoryCache, HybridCache, Redis).
- Answer trivially answerable questions (FAQ matches, simple commands) without a model.
- Deduplicate work in batch jobs.
```

### 6. Use batch processing for work that can wait

```plaintext
Provider batch APIs process large sets of requests asynchronously — typically with results
within hours and a meaningful discount versus real-time calls. Ideal for nightly
classification of records, bulk summarization, embedding large archives.
```

### 7. Measure, attribute, and alert

```csharp
// Turn token usage into money, using prices you keep in configuration
decimal cost =
    usage.InputTokenCount  / 1_000_000m * prices.InputPerMillion +
    usage.OutputTokenCount / 1_000_000m * prices.OutputPerMillion;

metrics.Record(feature: "support-chat", userId, model, usage, cost);
```

```plaintext
You can't optimize what you don't measure. Record tokens and estimated cost per feature,
per model, and ideally per user or tenant. Then:
  - Set provider-side budgets/spend alerts (and Azure budget alerts).
  - Alert on anomalies (a sudden jump in tokens per request is usually a bug or abuse).
  - Keep PRICES in configuration — they change, and hard-coded numbers go stale.
  - Run an evaluation set whenever you downgrade a model, to confirm quality held up.
```

### The order to attack it in

```plaintext
1. Measure first (log usage and cost).          4. Cache and route.
2. Right-size the model.                        5. Batch what can wait.
3. Cut prompt and history bloat; cap output.    6. Throttle users to bound the worst case.
```

---

## 13. A Provider-Neutral Option: Microsoft.Extensions.AI

### One abstraction over many providers

`Microsoft.Extensions.AI` defines a common `IChatClient` interface for chat models, so application code doesn't depend on any one vendor's SDK, and cross-cutting concerns (logging, caching, telemetry, rate limiting) can be added as **middleware**.

```csharp
using Microsoft.Extensions.AI;
using OpenAI;

IChatClient client = new OpenAIClient(apiKey)
    .GetChatClient("gpt-4o-mini")
    .AsIChatClient();                     // adapt the OpenAI ChatClient to the shared interface

ChatResponse response = await client.GetResponseAsync("Name three C# 12 features.");
Console.WriteLine(response.Text);
```

```csharp
// With dependency injection and middleware
builder.Services.AddChatClient(sp => new OpenAIClient(apiKey).GetChatClient("gpt-4o-mini").AsIChatClient())
    .UseLogging()
    .UseFunctionInvocation();             // automatically runs the tool-calling loop from Section 6
```

```plaintext
Benefits: swap providers (OpenAI, Azure OpenAI, local models) without rewriting calling
code; automatic tool-invocation loops; consistent logging/telemetry hooks; easier unit
testing against a fake IChatClient.

Trade-off: the abstraction exposes the COMMON features. Provider-specific capabilities
may still need the underlying SDK. Also — this library's API names changed during its
preview period (older samples use different method names), so match examples to the
package version you install.
```

---

## 14. Common Pitfalls

| Pitfall | Why it hurts | Better approach |
|---|---|---|
| API key in source control or client-side code | Anyone can extract it and run up charges on your account | Secrets store / Key Vault / Entra ID keyless auth; call the provider only from your backend (Section 1) |
| Creating a new client per request | Wastes connection setup; defeats reuse | Register the client as a singleton (Section 2) |
| Assuming the model remembers earlier calls | Each request is stateless; the model only sees what you send now | Store history yourself and resend it, trimmed to budget (Section 4, 8) |
| Ignoring `FinishReason` | A `Length` stop silently truncates answers and breaks JSON | Check it on every response and handle `Length`, `ContentFilter`, `ToolCalls` (Section 4) |
| Unbounded conversation history | Cost and latency grow every turn, until the context window overflows | Sliding window, summarization, or retrieval (Section 8) |
| Executing tool-call arguments without validation | Arguments come from a model steered by user input — a prompt-injection / injection risk | Validate and authorize server-side; use parameterized queries; confirm destructive actions (Section 6) |
| Letting a tool loop run forever | A confused model can request tools endlessly, burning tokens | Cap the number of rounds (Section 6) |
| Trusting structured output as *correct* | The schema guarantees shape, not truth; refusals and truncation can occur | Business-validate fields; check `Refusal` and `FinishReason` (Section 7) |
| Flushing nothing when streaming | The server buffers and users see no incremental text | Flush after every chunk; pass the cancellation token (Section 5) |
| Retrying every error | 400/401/403/404 and out-of-quota 429s will never succeed on retry | Retry only transient errors (408, rate-limit 429, 5xx) with jittered backoff (Section 10) |
| Retrying without jitter | Synchronized retries cause a thundering herd | Add random jitter; honor `Retry-After` (Section 10) |
| Un-gated parallel calls in batch jobs | A wall of concurrent requests triggers a wall of 429s | Bound concurrency with `SemaphoreSlim` or a rate limiter (Section 11) |
| Throttling by requests only | TPM can be exhausted by a few huge prompts | Track tokens as well as request counts (Section 11) |
| Unthrottled public AI endpoint | One abusive user or buggy loop can exhaust your quota and budget | Per-user rate limits and input-size caps on your own API (Section 11) |
| Using the biggest model for everything | Pays a large premium for tasks a small model handles fine | Route by task complexity; evaluate before downgrading (Section 12) |
| No usage logging or budget alerts | Cost surprises are discovered on the invoice | Log tokens/cost per feature; set spend alerts (Section 12) |
| Hard-coding model names and prices | Lineups and prices change; code goes stale | Keep model names and prices in configuration (Introduction, Section 12) |
| Tuning on one example prompt | Anecdotes mislead; changes may regress other cases | Keep a representative evaluation set and re-run it on every change (Section 9) |

---

## Quick Reference Table

| Concept | API / Technique | Purpose |
|---|---|---|
| OpenAI client | `new ChatClient(model, apiKey)` | Call OpenAI chat models |
| Azure OpenAI client | `new AzureOpenAIClient(uri, credential).GetChatClient(deployment)` | Same API against your Azure deployment |
| Keyless Azure auth | `new DefaultAzureCredential()` | Entra ID instead of stored keys |
| Basic call | `await client.CompleteChatAsync(messages, options, ct)` | Get a full response |
| Roles | `SystemChatMessage` / `UserChatMessage` / `AssistantChatMessage` / `ToolChatMessage` | Build the message list |
| Conversation memory | Resend history every call | Models are stateless |
| Why it stopped | `completion.FinishReason` | `Stop`, `Length`, `ToolCalls`, `ContentFilter` |
| Streaming | `CompleteChatStreamingAsync(messages)` + `await foreach` | Show text as it's generated |
| Tool definition | `ChatTool.CreateFunctionTool(name, description, schema)` | Describe a function to the model |
| Tool loop | Add `AssistantChatMessage`, run call, add `ToolChatMessage(call.Id, result)` | Let the model use your code |
| Structured output | `ChatResponseFormat.CreateJsonSchemaFormat(name, schema, strict: true)` | Constrain output to a JSON Schema |
| Token usage | `completion.Usage.InputTokenCount / OutputTokenCount` | Measure cost and context use |
| Count tokens | `TiktokenTokenizer.CreateForModel(model).CountTokens(text)` | Budget before sending |
| Temperature | `options.Temperature` | Randomness: low = focused, high = varied |
| Output cap | `options.MaxOutputTokenCount` | Bound length and cost |
| Retry settings | `OpenAIClientOptions.RetryPolicy` / `NetworkTimeout` | Tune built-in retries and timeouts |
| Error type | `ClientResultException` (`.Status`) | Branch on HTTP status |
| Client throttling | `TokenBucketRateLimiter`, `SemaphoreSlim` | Stay under RPM/TPM limits |
| Server throttling | `AddRateLimiter` + `RequireRateLimiting` | Protect your endpoint and budget |
| Caching | `IMemoryCache` / `HybridCache` / Redis | Skip repeat model calls |
| Provider-neutral | `IChatClient` (Microsoft.Extensions.AI) | Swap providers, add middleware |

---

## Conclusion

Integrating a hosted language model into a .NET application is easy to start and demanding to finish well, and almost every demanding part traces back to one of a few facts. The model is **stateless**, so you own the conversation and must keep it within a token budget. It is **probabilistic**, so structured outputs, validation, and low temperatures turn free-form text into something your code can trust — while never forgetting that a schema guarantees shape, not truth. It can only *propose* actions through tool calls, so your server remains the security boundary: validate every argument and authorize every effect. And it is a **metered, rate-limited remote service**, so retries with jitter, client-side throttling, per-user limits, caching, and cost telemetry are not extras but the difference between a demo and a dependable feature.

The same few habits carry across both OpenAI and Azure OpenAI, because the client design is shared: keep secrets out of code, reuse one client, pass cancellation tokens, stream for responsiveness, log token usage on every call, retry only what can succeed, and keep model names, prices, and limits in configuration where they can change without a redeploy. Get those right, and the genuinely hard questions — what to build, how to evaluate quality, and when a small model is good enough — become the main thing you spend your time on, exactly as they should be.

---

*Found this useful? Feel free to star the repo, open an issue with corrections, or share the "one runaway loop, one weekend, one surprise invoice" story that made the case for rate limits and spend alerts click better than any documentation page.*
