<script setup lang="ts">
import { useRouter } from "vue-router";

import { Thread } from "../api/threads";
import ChatInput from "../components/ChatInput.vue";
import ThreadListBar from "../components/ThreadListBar.vue";

const router = useRouter();

async function send(text: string): Promise<void> {
  if (text) {
    const thread = await Thread.create();
    router.push({ path: `/threads/${thread.id}`, query: { initial: text } });
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
