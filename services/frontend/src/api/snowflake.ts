import { snowflakeTime } from "./utils";

export default class Snowflake {
  public readonly id: bigint;

  public constructor(id: bigint) {
    this.id = id;
  }

  public get createdAt(): Date {
    return snowflakeTime(this.id);
  }

  public static parse(data: Snowflake): Snowflake {
    return new Snowflake(BigInt(data.id));
  }

  public get renderCreatedAt(): string {
    const now = new Date();
    if (now.getFullYear() == this.createdAt.getFullYear()
      && now.getMonth() == this.createdAt.getMonth()
    ) {
      if (now.getDate() == this.createdAt.getDate()) {
        return `Today at ${this.createdAt.toLocaleTimeString()}`;
      }

      if (now.getDate() - this.createdAt.getDate() == 1) {
        return `Yesterday at ${this.createdAt.toLocaleTimeString()}`;
      }
    }

    return this.createdAt.toLocaleString();
  }
}