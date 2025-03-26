import { Converter } from "showdown";

const EPOCH = BigInt(Date.UTC(2025, 0, 1, 0, 0, 0, 0));

export function snowflakeTime(id: bigint): Date {
  return new Date(Number(EPOCH + (id >> 16n)));
}

export const MARKDOWN = new Converter();
