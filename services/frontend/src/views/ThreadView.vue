<script setup lang="ts">
import { ref } from "vue";
import { onBeforeRouteUpdate, useRoute } from "vue-router";

import { LangChainMessage, Message, Thread } from "../api/threads";
import ChatInput from "../components/ChatInput.vue";
import ThreadListBar from "../components/ThreadListBar.vue";
import MessageTile from "../components/MessageTile.vue";
import useStore from "../store/index";

const route = useRoute();
const store = useStore();

class PageState {
  public readonly history = ref([] as Message[]);
  public readonly streamMessage = ref(null as Message | null);

  public thread: Thread | null = null;
  public streaming: boolean = false;

  public constructor(id?: string) {
    Thread.fetch(BigInt(id ?? route.params.id as string)).then(
      async (t) => {
        this.thread = t;
        await fetchHistory();

        const initial = store.consumeInitialMessage();
        if (initial !== null && this.history.value.length === 0) {
          send(initial.input, initial.file);
        }
      },
    );
  }
}

async function fetchHistory(): Promise<void> {
  const thread = state.thread;
  if (thread) {
    const m = await thread.history();
    // state.history.value = m.map((x) => x.valueOf());
    state.history.value = m;
  }
}

async function send(text: string, file: File | null): Promise<void> {
  if (state.thread && text) {
    const sse = await state.thread.send(text, file, store.$state.temperature);
    sse.addEventListener(
      "ai",
      (e: any) => {  // eslint-disable-line @typescript-eslint/no-explicit-any
        const m = Message.parse(JSON.parse(e.data));
        state.streamMessage.value = null;
        state.history.value.unshift(m);
      });
    sse.addEventListener(
      "chunk",
      (e: any) => {  // eslint-disable-line @typescript-eslint/no-explicit-any
        const thread = state.thread;
        if (thread) {
          let content = state.streamMessage.value?.data.content ?? "";
          const newContent = JSON.parse(e.data).content;
          if (state.streaming) {
            state.streaming = false;
            content = newContent;
          } else {
            content += newContent;
          }

          state.streamMessage.value = new Message(0n, new LangChainMessage(content, "ai"), null, thread);
        }
      }
    );
    sse.addEventListener(
      "event",
      (e: any) => {  // eslint-disable-line @typescript-eslint/no-explicit-any
        state.streaming = true;
        const thread = state.thread;
        if (thread) {
          state.streamMessage.value = new Message(0n, new LangChainMessage(e.data, "ai"), null, thread);
        }
      }
    );
  }
}

let state = new PageState();
onBeforeRouteUpdate(
  (to) => {
    state = new PageState(to.params.id as string);
  },
);
</script>

<template>
  <ThreadListBar :key="useRoute().params.id as string">
    <div class="main d-grid h-100 w-100">
      <div class="d-flex flex-column h-100 w-100">
        <div class="flex-grow-1 overflow-y-scroll px-1 w-100">
          <MessageTile v-for="m in state.history.value.slice().reverse()" :message="m" :key="m.id.toString()" />
          <div v-if="state.streamMessage.value">
            <MessageTile :message="state.streamMessage.value"></MessageTile>
          </div>
        </div>
        <div class="px-3 py-3 w-100">
          <ChatInput :onSubmit="send"></ChatInput>
        </div>
      </div>
      <div class="h-100 p-2 text-white w-100">
        <h5>Model settings</h5>
        <div class="separator"></div>
        <h6>Temperature</h6>
        <div class="d-flex flex-row">
          <input class="d-block flex-grow-1 p-1" type="range" min="0" max="2" step="0.1" v-model="store.$state.temperature">
          <div class="separator"></div>
          <span class="d-block">{{ store.$state.temperature }}</span>
        </div>
      </div>
    </div>
  </ThreadListBar>
</template>

<style lang="css" scoped>
.main {
  grid-template-columns: 1fr 0;
  grid-template-rows: 100%;
}

.separator {
  height: 1rem;
  width: 1rem;
}

@media (min-width: 992px) {
  .main {
    grid-template-columns: 1fr 300px;
  }
}
</style>
