using System.Net.Http.Json;
using System.Text;
using System.Text.Json;

const string LlmModel = "llama3.2:1b";
const string EmbeddingModel = "nomic-embed-text";

using var http = new HttpClient
{
    BaseAddress = new Uri("http://localhost:11434")
};

// ------------------------------------------------------------
// 1. Our tiny knowledge base
// ------------------------------------------------------------

var document = """
Acme Corporation

Acme was founded in 2018 and is headquartered in Vancouver.

The company has three main products:
Acme Cloud, Acme Analytics, and Acme Secure.

Acme Cloud is a cloud storage platform designed for small businesses.

Acme Analytics provides dashboards and reporting capabilities.

Acme Secure provides identity and access management.

The company currently has 120 employees.

The CEO is Jane Smith.

Acme plans to launch its European operations in 2027.
""";

// ------------------------------------------------------------
// 2. Split document into chunks
// ------------------------------------------------------------

var chunks = document
    .Split("\n\n", StringSplitOptions.RemoveEmptyEntries)
    .Select(x => x.Trim())
    .Where(x => x.Length > 0)
    .ToList();

Console.WriteLine("Creating embeddings...");

var vectorStore = new List<(string Text, float[] Vector)>();

foreach (var chunk in chunks)
{
    var vector = await GetEmbedding(chunk);
    vectorStore.Add((chunk, vector));
}

Console.WriteLine($"Indexed {vectorStore.Count} chunks.");
Console.WriteLine();
Console.WriteLine("RAG demo ready.");
Console.WriteLine("Ask a question, or type 'exit'.");
Console.WriteLine();

// ------------------------------------------------------------
// 3. Question → Retrieval → LLM
// ------------------------------------------------------------

while (true)
{
    Console.Write("Question: ");
    var question = Console.ReadLine();

    if (string.IsNullOrWhiteSpace(question))
        continue;

    if (question.Equals("exit", StringComparison.OrdinalIgnoreCase))
        break;

    // Embed the user's question
    var questionVector = await GetEmbedding(question);

    // Find most similar document chunks
    var relevantChunks = vectorStore
        .Select(x => new
        {
            x.Text,
            Score = CosineSimilarity(questionVector, x.Vector)
        })
        .OrderByDescending(x => x.Score)
        .Take(3)
        .ToList();

    Console.WriteLine();
    Console.WriteLine("Retrieved context:");

    foreach (var chunk in relevantChunks)
    {
        Console.WriteLine(
            $"[{chunk.Score:F3}] {chunk.Text}");
    }

    // Build context for the LLM
    var context = string.Join(
        "\n\n",
        relevantChunks.Select(x => x.Text));

    var prompt = $"""
You are answering questions about Acme Corporation.

Use ONLY the information in the CONTEXT.

If the answer cannot be found in the context,
say: "I don't have that information in the provided document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""";

    var answer = await AskLlm(prompt);

    Console.WriteLine();
    Console.WriteLine("LLM answer:");
    Console.WriteLine(answer);
    Console.WriteLine();
}

// ------------------------------------------------------------
// Embeddings
// ------------------------------------------------------------

async Task<float[]> GetEmbedding(string text)
{
    var response = await http.PostAsJsonAsync(
        "/api/embed",
        new
        {
            model = EmbeddingModel,
            input = text
        });

    response.EnsureSuccessStatusCode();

    var result =
        await response.Content.ReadFromJsonAsync<EmbeddingResponse>();

    return result!.embeddings[0];
}

// ------------------------------------------------------------
// LLM
// ------------------------------------------------------------

async Task<string> AskLlm(string prompt)
{
    var response = await http.PostAsJsonAsync(
        "/api/chat",
        new
        {
            model = LlmModel,
            messages = new[]
            {
                new
                {
                    role = "user",
                    content = prompt
                }
            },
            stream = false
        });

    response.EnsureSuccessStatusCode();

    var result =
        await response.Content.ReadFromJsonAsync<ChatResponse>();

    return result!.message.content;
}

// ------------------------------------------------------------
// Cosine similarity
// ------------------------------------------------------------

static double CosineSimilarity(float[] a, float[] b)
{
    double dot = 0;
    double magnitudeA = 0;
    double magnitudeB = 0;

    for (int i = 0; i < a.Length; i++)
    {
        dot += a[i] * b[i];
        magnitudeA += a[i] * a[i];
        magnitudeB += b[i] * b[i];
    }

    if (magnitudeA == 0 || magnitudeB == 0)
        return 0;

    return dot /
           (Math.Sqrt(magnitudeA) * Math.Sqrt(magnitudeB));
}

// ------------------------------------------------------------
// Ollama response models
// ------------------------------------------------------------

record EmbeddingResponse(float[][] embeddings);

record ChatResponse(ChatMessage message);

record ChatMessage(string role, string content);