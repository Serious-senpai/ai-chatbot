import { defineStore } from "pinia";

interface InitialMessage {
  input: string;
  file: File | null;
}

const useStore = defineStore(
  "store",
  {
    state: () => ({
      initial: null as InitialMessage | null,
    }),
    actions: {
      setInitialMessage(input: string, file: File | null): void {
        this.initial = {
          input, file
        };
      },

      consumeInitialMessage(): InitialMessage | null {
        const message = this.initial;
        this.initial = null;
        return message;
      }
    },
  }
);

export default useStore;