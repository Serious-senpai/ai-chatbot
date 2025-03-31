<script setup lang="ts">
import { useRouter } from "vue-router";

import { Thread } from "../api/threads";
import ChatInput from "../components/ChatInput.vue";
import ThreadListBar from "../components/ThreadListBar.vue";
import useStore from "../store/index";

const router = useRouter();
const store = useStore();

async function send(input: string, file: File | null): Promise<void> {
  if (input) {
    const thread = await Thread.create();
    store.setInitialMessage(input, file);
    router.push(`/threads/${thread.id}`);
  }
}
</script>

<template>
  <ThreadListBar>
    <div class="position-relative start-50 w-50 top-50 translate-middle">
      <h1 class="m-4 text-center text-white">What can I help you?</h1>
      <ChatInput :onSubmit="send"></ChatInput>
    </div>
  </ThreadListBar>
</template>
