<script setup lang="ts">
import { ref } from "vue";

import { Message } from "../api/threads";
import { MARKDOWN } from "../api/utils";

const props = defineProps(
  {
    message: {
      type: Message,
      required: true,
    },
  },
);

const isHuman = ref(props.message.data.type === "human");
</script>

<template>
  <div class="w-100" :class="isHuman ? ['text-end'] : []">
    <div class="d-flex align-items-baseline" :class="isHuman ? ['flex-row-reverse'] : []">
      <span class="fs-6 text-info">{{ message.renderAuthor }}</span>
      <span class="time text-secondary">{{ message.renderCreatedAt }}</span>
    </div>
    <div class="fs-6 text-white" v-html="MARKDOWN.makeHtml(message.renderContent)">
    </div>
  </div>
</template>

<style lang="css" scoped>
.time {
  font-size: 0.7rem;
  margin: 0 0.5rem;
}
</style>
