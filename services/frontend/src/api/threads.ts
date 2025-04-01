import JSONBigInt from "json-bigint";
import { SSE } from "sse.js";

import client from "./client";
import Snowflake from "./snowflake";

export class Thread extends Snowflake {
  public async history(): Promise<Message[]> {
    const response = await client.get<string>(
      `/chat/${this.id}/messages`,
      { transformResponse: [data => data] },
    );

    const messages = JSONBigInt.parse(response.data) as Message[];
    return messages.map(Message.parse);
  }

  public async send(content: string, file: File | null, temperature: number): Promise<SSE> {
    const uri = client.getUri({ url: `/chat/${this.id}/messages` });
    const data = {
      content: content,
      file: await new Promise(
        (resolve, reject) => {
          if (file !== null) {
            const reader = new FileReader();
            reader.onload = (e: ProgressEvent<FileReader>) => {
              let b64 = e.target?.result;
              if (typeof b64 === "string") {
                b64 = b64.split(",")[1];
                resolve({
                  filename: file.name,
                  data: b64
                });
              } else {
                reject(new Error("Failed to read file"));
              }
            };
            reader.onerror = (e: ProgressEvent<FileReader>) => {
              reject(e);
            };

            reader.readAsDataURL(file);
          } else {
            resolve(null);
          }
        }
      ),
      temperature,
    };

    return new SSE(
      uri,
      {
        headers: {
          "Content-Type": "application/json",
        },
        payload: JSON.stringify(data),
      },
    );
  }

  public static parse(data: Thread): Thread {
    return new Thread(BigInt(data.id));
  }

  public static async query(): Promise<Thread[]> {
    const response = await client.get<string>(
      "/chat",
      { transformResponse: [data => data] },
    );

    const threads = JSONBigInt.parse(response.data) as Thread[];
    return threads.sort((a, b) => Number(a.id - b.id)).map(Thread.parse);
  }

  public static async fetch(id: bigint): Promise<Thread> {
    const response = await client.get<string>(
      `/chat/${id}`,
      { transformResponse: [data => data] },
    );

    return Thread.parse(JSONBigInt.parse(response.data));
  }

  public static async create(): Promise<Thread> {
    const response = await client.post<string>(
      "/chat",
      {},
      { transformResponse: [data => data] },
    );

    return Thread.parse(JSONBigInt.parse(response.data));
  }
}

export class LangChainMessage {
  public readonly content: string;
  public readonly type: string;

  public constructor(content: string, type: string) {
    this.content = content;
    this.type = type;
  }

  public static parse(data: LangChainMessage): LangChainMessage {
    return new LangChainMessage(data.content, data.type);
  }
}

export class Message extends Snowflake {
  public readonly data: LangChainMessage;
  public readonly attachment: string | null;
  public readonly thread: Thread;

  public constructor(
    id: bigint,
    data: LangChainMessage,
    attachment: string | null,
    thread: Thread,
  ) {
    super(id);
    this.data = data;
    this.attachment = attachment;
    this.thread = thread;
  }

  public static parse(data: Message): Message {
    return new Message(
      BigInt(data.id),
      LangChainMessage.parse(data.data),
      data.attachment,
      Thread.parse(data.thread),
    );
  }

  public get renderAuthor(): string {
    switch (this.data.type) {
      case "human":
        return "Human";

      case "ai":
        return "AI";

      case "tool":
        return "Tool";

      default:
        return this.data.type;
    }
  }

  public get renderContent(): string {
    switch (this.data.type) {
      case "tool":
        return "```\n" + this.data.content + "\n```";

      default:
        return this.data.content;
    }
  }
}
