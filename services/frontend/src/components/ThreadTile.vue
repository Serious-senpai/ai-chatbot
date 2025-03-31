<script setup lang="ts">
import { RouterLink } from "vue-router";

import { Thread } from "../api/threads";

defineProps(
  {
    thread: {
      type: Thread,
      required: true,
    },
  },
);
</script>

<template>
  <RouterLink class="bar d-block p-1 rounded-2 text-decoration-none text-end text-white w-100" :to="`/threads/${thread.id}`">
    {{ thread.renderCreatedAt }}
  </RouterLink>
</template>

<style lang="css" scoped>
.bar:hover {
  background-color: gray;
}
</style>