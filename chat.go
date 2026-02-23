package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

func main() {
	verbose := flag.Bool("verbose", false, "enable verbose logging")
	flag.Parse()

	endpoint := os.Getenv("AZURE_AI_FOUNDRY_ENDPOINT")
	apiKey := os.Getenv("AZURE_AI_FOUNDRY_KEY")
	model := os.Getenv("AZURE_DEPLOYMENT_NAME")
	if model == "" {
		model = "claude-sonnet-4-6"
	}

	client := anthropic.NewClient(
		option.WithBaseURL(endpoint+"/anthropic/"),
		option.WithAPIKey(apiKey),
		option.WithDefaultHeader("x-api-key", apiKey),
		option.WithDefaultHeader("anthropic-version", "2023-06-01"),
	)

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	agent := NewAgent(&client, getUserMessage, model, *verbose)
	if err := agent.Run(context.TODO()); err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

func NewAgent(client *anthropic.Client, getUserMessage func() (string, bool), model string, verbose bool) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
		model:          model,
		verbose:        verbose,
	}
}

type Agent struct {
	client         *anthropic.Client
	getUserMessage func() (string, bool)
	model          string
	verbose        bool
}

func (a *Agent) Run(ctx context.Context) error {
	var conversation []anthropic.MessageParam

	fmt.Println("Chat with Claude on Azure | 'ctrl-c' to quit")

	for {
		fmt.Print("\u001b[94mYou\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok || userInput == "" {
			break
		}

		conversation = append(conversation, anthropic.NewUserMessage(anthropic.NewTextBlock(userInput)))

		if a.verbose {
			fmt.Printf("[verbose] sending %d messages to model %s\n", len(conversation), a.model)
		}

		resp, err := a.client.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     a.model,
			MaxTokens: 1024,
			Messages:  conversation,
		})
		if err != nil {
			return fmt.Errorf("inference error: %w", err)
		}

		assistantText := resp.Content[0].Text
		fmt.Printf("\u001b[93mClaude\u001b[0m: %s\n", assistantText)

		conversation = append(conversation, anthropic.NewAssistantMessage(anthropic.NewTextBlock(assistantText)))
	}

	return nil
}
